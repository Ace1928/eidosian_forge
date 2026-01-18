import math
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn.utils.distributed import get_dim_for_local_rank
class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (super().forward(input), input)