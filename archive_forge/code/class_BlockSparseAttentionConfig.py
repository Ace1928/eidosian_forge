import logging
import math
from dataclasses import dataclass
import torch
from xformers import _is_triton_available
from xformers.components.attention import Attention, AttentionConfig, register_attention
@dataclass
class BlockSparseAttentionConfig(AttentionConfig):
    layout: torch.Tensor
    block_size: int
    dropout: float
    num_heads: int