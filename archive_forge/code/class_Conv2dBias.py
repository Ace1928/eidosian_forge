from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
class Conv2dBias(nn.Module):
    """Model with only Conv2d layers, some with bias, some in a Sequential and some outside.
    Used to test pruned Conv2d-Bias-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(1, 32, 3, 1, bias=True), nn.Conv2d(32, 32, 3, 1, bias=True), nn.Conv2d(32, 64, 3, 1, bias=False))
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=True)
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        return x