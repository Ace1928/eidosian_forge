from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
class LinearBias(nn.Module):
    """Model with only Linear layers, alternating layers with biases,
    wrapped in a Sequential. Used to test pruned Linear-Bias-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(7, 5, bias=True), nn.Linear(5, 6, bias=False), nn.Linear(6, 3, bias=True), nn.Linear(3, 3, bias=True), nn.Linear(3, 10, bias=False))

    def forward(self, x):
        x = self.seq(x)
        return x