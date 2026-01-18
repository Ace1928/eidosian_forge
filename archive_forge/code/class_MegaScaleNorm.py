import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
class MegaScaleNorm(nn.Module):
    """
    Scale normalization introduced in MEGA which is similar to RMSNorm, but uses a single parameter for scalar
    multiplication instead of a vector, and applies over a specified dimension
    """

    def __init__(self, dim, eps=1e-06, affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.scalar = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('scalar', None)

    def forward(self, input):
        mean_square = torch.mean(torch.square(input), dim=self.dim, keepdim=True)
        if self.scalar is not None:
            input = self.scalar * input
        output = input * torch.rsqrt(mean_square + self.eps)
        return output