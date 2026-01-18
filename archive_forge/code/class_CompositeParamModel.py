from typing import Tuple
import torch
import torch.nn as nn
class CompositeParamModel(nn.Module):

    def __init__(self, device: torch.device):
        super().__init__()
        self.l = nn.Linear(100, 100, device=device)
        self.u1 = UnitModule(device)
        self.u2 = UnitModule(device)
        self.p = nn.Parameter(torch.randn((100, 100), device=device))
        self.register_buffer('buffer', torch.randn((100, 100), device=device), persistent=True)

    def forward(self, x):
        a = self.u2(self.u1(self.l(x)))
        b = self.p
        return torch.mm(a, b)