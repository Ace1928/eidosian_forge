from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
class SimpleConv2d(nn.Module):
    """Model with only Conv2d layers, all without bias, some in a Sequential and some following.
    Used to test pruned Conv2d-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(1, 32, 3, 1, bias=False), nn.Conv2d(32, 64, 3, 1, bias=False))
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=False)
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        return x