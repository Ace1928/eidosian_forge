from typing import Tuple
import torch
import torch.nn as nn
class UnitParamModule(nn.Module):

    def __init__(self, device: torch.device):
        super().__init__()
        self.l = nn.Linear(100, 100, device=device)
        self.seq = nn.Sequential(nn.ReLU(), nn.Linear(100, 100, device=device), nn.ReLU())
        self.p = nn.Parameter(torch.randn((100, 100), device=device))

    def forward(self, x):
        return torch.mm(self.seq(self.l(x)), self.p)