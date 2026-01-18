from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import scaled_dot_product_attention
@dataclass
class LinformerSelfAttentionConfig(AttentionConfig):
    seq_len: int
    k: Optional[int]