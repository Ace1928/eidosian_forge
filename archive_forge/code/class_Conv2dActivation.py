from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
class Conv2dActivation(nn.Module):
    """Model with only Conv2d layers, some with bias, some in a Sequential and some following.
    Activation function modules in between each Sequential layer, functional activations called
    in-between each outside layer.
    Used to test pruned Conv2d-Bias-Activation-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(1, 32, 3, 1, bias=True), nn.ReLU(), nn.Conv2d(32, 64, 3, 1, bias=True), nn.Tanh(), nn.Conv2d(64, 64, 3, 1, bias=False), nn.ReLU())
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=False)
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=True)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.relu(x)
        x = self.conv2d2(x)
        x = F.hardtanh(x)
        return x