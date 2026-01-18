import logging
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import (
from xformers.components.attention.utils import (
def _triu_mask(self, dim_1: int, dim_2: int, dim_3: int, **kwargs) -> torch.Tensor:
    device = kwargs['device']
    dtype = kwargs['dtype']
    return torch.triu(torch.ones(dim_2, dim_3, dtype=dtype, device=device) * float('-inf'), diagonal=1).expand(dim_1, -1, -1)