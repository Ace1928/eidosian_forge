import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
from torch import nn
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
class FeedForward(nn.Module):
    """
    A feedforward neural network module that applies a sequence of transformations.

    Attributes:
        w1 (ColumnParallelLinear): First linear transformation with column parallelism.
        w2 (RowParallelLinear): Second linear transformation with row parallelism.
        w3 (ColumnParallelLinear): Third linear transformation with column parallelism.

    Methods:
        forward(x): Applies the feedforward network transformations to the input tensor.
    """

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]=None):
        """
        Initializes the FeedForward module with specified dimensions and multipliers.

        Args:
            dim (int): Dimensionality of the input and output.
            hidden_dim (int): Base dimensionality of the hidden layers.
            multiple_of (int): Ensures hidden dimensions are multiples of this value.
            ffn_dim_multiplier (Optional[float]): Optional multiplier for the hidden dimension.
        """
        super().__init__()
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        else:
            hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeedForward module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the feedforward transformations.
        """
        x1 = self.w1(x)
        x2 = self.w3(x)
        x1_activated = F.silu(x1)
        x3 = x1_activated * x2
        output = self.w2(x3)
        return output