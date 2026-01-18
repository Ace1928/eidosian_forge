from enum import Enum
from typing import Optional
import torch
from torch import nn
class StarReLU(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = torch.nn.functional.relu(x)
        return 0.8944 * x_ * x_ - 0.4472