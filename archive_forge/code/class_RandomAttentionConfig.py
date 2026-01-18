from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn as nn
from xformers.components.attention import (
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import scaled_dot_product_attention
@dataclass
class RandomAttentionConfig(AttentionConfig):
    r: Optional[float]
    constant_masking: Optional[bool]
    force_sparsity: Optional[bool]