import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vendor_packages'))
import torch
import torch.nn as nn
import numpy as np
import cv2
import timm

label = ['cloudy', 'rainy', 'snowy', 'sunny']
im_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
std = np.array([0.5, 0.5, 0.5], dtype=np.float32)


class WeatherCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnetv2_m.in21k_ft_in1k',
            pretrained=False,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


model = WeatherCNN(num_classes=len(label)).to(device)
state_dict = torch.load("./results/model_sample.pth", map_location=device)
if not any(k.startswith('model.') for k in state_dict.keys()):
    state_dict = {f'model.{k}': v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()


def predict(X):
    """
    模型预测
    param：
        X : np.ndarray，由 cv2.imread 读取的图片数据，shape(224,224,3)。
    return：
        y_predict : str, 天气类别标签，取值为 'sunny', 'cloudy', 'rainy', 'snowy' 之一。
    """
    X = cv2.resize(X, (im_size, im_size))
    X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
    X = X.astype(np.float32) / 255.0
    X = (X - mean) / std
    # HWC -> CHW，加 batch 维
    X = np.transpose(X, (2, 0, 1))[np.newaxis, :, :, :]
    X = torch.from_numpy(X).to(device)

    with torch.no_grad():
        prediction = model(X)
    y_predict = label[int(torch.argmax(prediction, dim=1).item())]
    return y_predict
