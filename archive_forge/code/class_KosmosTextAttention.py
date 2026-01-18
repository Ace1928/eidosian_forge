import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_kosmos2 import Kosmos2Config, Kosmos2TextConfig, Kosmos2VisionConfig
class KosmosTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, add_inner_attn_layernorm: bool=False, bias: bool=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.inner_attn_ln = None
        if add_inner_attn_layernorm:
            self.inner_attn_ln = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def _shape(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, self.head_dim)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = encoder_hidden_states is not None
        batch_size, seq_length = hidden_states.shape[:2]
        current_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if is_cross_attention and past_key_value and (past_key_value[0].shape[2] == current_states.shape[1]):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else:
            key_states = self._shape(self.k_proj(current_states))
            value_states = self._shape(self.v_proj(current_states))
            if past_key_value is not None and (not is_cross_attention):
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
        query_states = self._shape(self.q_proj(hidden_states) * self.scaling)
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        if self.is_decoder:
            past_key_value = (key_states, value_states)
        src_len = key_states.size(2)
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, seq_length, src_len):
                raise ValueError(f'Attention mask should be of size {(batch_size, 1, seq_length, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        context_states = torch.matmul(attn_weights, value_states)
        context_states = context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        if self.inner_attn_ln is not None:
            context_states = self.inner_attn_ln(context_states)
        attn_output = self.out_proj(context_states)
        return (attn_output, attn_weights, past_key_value)