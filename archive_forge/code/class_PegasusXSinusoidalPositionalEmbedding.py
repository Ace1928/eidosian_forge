import dataclasses
import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_pegasus_x import PegasusXConfig
class PegasusXSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, embed_dim, max_scale: int=10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_scale = max_scale

    @torch.no_grad()
    def forward(self, input_embeds: torch.Tensor, past_key_values_length: int=0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        batch_size, seq_len = input_embeds.shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=input_embeds.device)[:, None]
        pe = torch.zeros((seq_len, self.embed_dim), device=input_embeds.device, dtype=input_embeds.dtype)
        half_d_feature = self.embed_dim // 2
        div_term = torch.exp(torch.arange(half_d_feature, device=input_embeds.device, dtype=torch.int64).type_as(input_embeds) * -(np.log(float(self.max_scale)) / (half_d_feature - 1)))
        pe[:, :half_d_feature] = torch.sin(positions * div_term)
        pe[:, half_d_feature:] = torch.cos(positions * div_term)
        return pe[None].expand(batch_size, -1, -1)