from typing import List
import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from xformers.components import RequiresWrappedInputs
class ReversibleBlock(nn.Module):

    def __init__(self, f: nn.Module, g: nn.Module, split_dim: int=-1):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.split_dim = split_dim

    def forward(self, x: torch.Tensor, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1, y2 = (None, None)
        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)
        return torch.cat([y1, y2], dim=self.split_dim)

    def backward_pass(self, y: torch.Tensor, dy: torch.Tensor, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=self.split_dim)
        del y
        dy1, dy2 = torch.chunk(dy, 2, dim=self.split_dim)
        del dy
        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)
        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None
        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1)
        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None
            x = torch.cat([x1, x2.detach()], dim=self.split_dim)
            dx = torch.cat([dx1, dx2], dim=self.split_dim)
        return (x, dx)