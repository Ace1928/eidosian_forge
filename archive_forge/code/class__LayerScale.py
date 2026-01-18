import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
class _LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float=0):
        """
        Args:
            channels (int): Size of  rescaling
            init (float, optional): Scale to default to (default: 0)
        """
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LayerScale forward call

        Args:
            x (torch.Tensor): input tensor for LayerScale

        Returns:
            Tensor
                Output after rescaling tensor.
        """
        return self.scale[:, None] * x