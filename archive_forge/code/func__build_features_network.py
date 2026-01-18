import math
import torch
def _build_features_network():
    layers = []
    for input_dim, output_dim in [(1, 64), (64, 128)]:
        layers += [torch.nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)]
    for input_dim, output_dim in [(128, 256), (256, 512)]:
        layers += [torch.nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)]
    return torch.nn.Sequential(*layers)