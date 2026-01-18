from typing import Tuple
import torch
import torch.nn as nn
class UnitModule(nn.Module):

    def __init__(self, device: torch.device):
        super().__init__()
        self.l1 = nn.Linear(100, 100, device=device)
        self.seq = nn.Sequential(nn.ReLU(), nn.Linear(100, 100, device=device), nn.ReLU())
        self.l2 = nn.Linear(100, 100, device=device)

    def forward(self, x):
        return self.l2(self.seq(self.l1(x)))