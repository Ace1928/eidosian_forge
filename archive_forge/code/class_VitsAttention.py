import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig
class VitsAttention(nn.Module):
    """Multi-headed attention with relative positional representation."""

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.window_size = config.window_size
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** (-0.5)
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.embed_dim} and `num_attention_heads`: {self.num_heads}).')
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        if self.window_size:
            self.emb_rel_k = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)
            self.emb_rel_v = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}')
        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, src_len)
            relative_logits = torch.matmul(query_states, key_relative_embeddings.transpose(-2, -1))
            rel_pos_bias = self._relative_position_to_absolute_position(relative_logits)
            attn_weights += rel_pos_bias
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}')
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        if self.window_size is not None:
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, src_len)
            relative_weights = self._absolute_position_to_relative_position(attn_probs)
            rel_pos_bias = torch.matmul(relative_weights, value_relative_embeddings)
            attn_output += rel_pos_bias
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped)

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        if pad_length > 0:
            relative_embeddings = nn.functional.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])
        slice_start_position = max(self.window_size + 1 - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        return relative_embeddings[:, slice_start_position:slice_end_position]

    def _relative_position_to_absolute_position(self, x):
        batch_heads, length, _ = x.size()
        x = nn.functional.pad(x, [0, 1, 0, 0, 0, 0])
        x_flat = x.view([batch_heads, length * 2 * length])
        x_flat = nn.functional.pad(x_flat, [0, length - 1, 0, 0])
        x_final = x_flat.view([batch_heads, length + 1, 2 * length - 1])
        x_final = x_final[:, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch_heads, length, _ = x.size()
        x = nn.functional.pad(x, [0, length - 1, 0, 0, 0, 0])
        x_flat = x.view([batch_heads, length * (2 * length - 1)])
        x_flat = nn.functional.pad(x_flat, [length, 0, 0, 0])
        x_final = x_flat.view([batch_heads, length, 2 * length])[:, :, 1:]
        return x_final