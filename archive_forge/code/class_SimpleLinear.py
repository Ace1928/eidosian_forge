from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
class SimpleLinear(nn.Module):
    """Model with only Linear layers without biases, some wrapped in a Sequential,
    some following the Sequential. Used to test basic pruned Linear-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(7, 5, bias=False), nn.Linear(5, 6, bias=False), nn.Linear(6, 4, bias=False))
        self.linear1 = nn.Linear(4, 4, bias=False)
        self.linear2 = nn.Linear(4, 10, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x