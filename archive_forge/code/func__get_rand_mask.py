from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn as nn
from xformers.components.attention import (
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import scaled_dot_product_attention
def _get_rand_mask(self, shape: torch.Size) -> torch.Tensor:
    sparsity = 1 - self.r
    mask = random_pattern(shape[1], sparsity=sparsity)
    if self.causal:
        mask &= causal_1d_pattern(shape[1])
    mask = sparsify(mask) if self.force_sparsity else maybe_sparsify(mask)
    return mask