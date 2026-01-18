import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class RangeSigmoid(nn.Module):

    def __init__(self, val_range: Tuple[float, float]=(0.0, 1.0)) -> None:
        super(RangeSigmoid, self).__init__()
        assert isinstance(val_range, tuple) and len(val_range) == 2
        self.val_range: Tuple[float, float] = val_range
        self.sigmoid: nn.modules.Module = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.sigmoid(x) * (self.val_range[1] - self.val_range[0]) + self.val_range[0]
        return out