import logging
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.init import constant_
from xformers.components.attention import Attention
from xformers.components.input_projection import InputProjection, InputProjectionConfig
from xformers.components.positional_embedding import RotaryEmbedding
def _fold_heads(t: torch.Tensor, B: int, S: int, H: int, Hs: int):
    return t.view(B, S, H, Hs).transpose(1, 2).flatten(start_dim=0, end_dim=1)