import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
class ScalingLayer(nn.Module):
    """Scaling layer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None], persistent=False)
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None], persistent=False)

    def forward(self, inp: Tensor) -> Tensor:
        """Process input."""
        return (inp - self.shift) / self.scale