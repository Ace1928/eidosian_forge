import math
import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_olmo import OlmoConfig
class OlmoDynamicNTKScalingRotaryEmbedding(OlmoRotaryEmbedding):
    """OlmoRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            self.register_buffer('inv_freq', inv_freq, persistent=False)
        cos, sin = super().forward(x, position_ids)
        return (cos, sin)