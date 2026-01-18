import math
from dataclasses import dataclass
from typing import (
import torch
def add_bias(self, bias: torch.Tensor) -> 'LowerTriangularMaskWithTensorBias':
    """
        Creates a new causal mask with an arbitrary ``torch.Tensor`` bias
        """
    return LowerTriangularMaskWithTensorBias(bias)