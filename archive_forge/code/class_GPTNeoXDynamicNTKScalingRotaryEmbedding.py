from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10, logging
from .configuration_gpt_neox import GPTNeoXConfig
class GPTNeoXDynamicNTKScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
            self.register_buffer('inv_freq', inv_freq, persistent=False)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)