import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_autoformer import AutoformerConfig
class AutoformerAttention(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
        (1) period-based dependencies discovery (2) time delay aggregation
    This block replace the canonical self-attention mechanism.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True, autocorrelation_factor: int=3):
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
        self.autocorrelation_factor = autocorrelation_factor

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        if is_cross_attention and past_key_value is not None and (past_key_value[0].shape[2] == key_value_states.shape[1]):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = (key_states, value_states)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        queries_time_length = query_states.size(1)
        values_time_length = value_states.size(1)
        if queries_time_length > values_time_length:
            query_states = query_states[:, :queries_time_length - values_time_length, :]
            zeros = torch.zeros_like(query_states).float()
            value_states = torch.cat([value_states, zeros], dim=1)
            key_states = torch.cat([key_states, zeros], dim=1)
        else:
            value_states = value_states[:, :queries_time_length, :]
            key_states = key_states[:, :queries_time_length, :]
        query_states_fft = torch.fft.rfft(query_states, n=tgt_len, dim=1)
        key_states_fft = torch.fft.rfft(key_states, n=tgt_len, dim=1)
        attn_weights = query_states_fft * torch.conj(key_states_fft)
        attn_weights = torch.fft.irfft(attn_weights, n=tgt_len, dim=1)
        src_len = key_states.size(1)
        channel = key_states.size(2)
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, channel):
            raise ValueError(f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, channel)}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}')
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, channel)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, channel)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, channel)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, channel)
        else:
            attn_weights_reshaped = None
        time_length = value_states.size(1)
        autocorrelations = attn_weights.view(bsz, self.num_heads, tgt_len, channel)
        top_k = int(self.autocorrelation_factor * math.log(time_length))
        autocorrelations_mean_on_head_channel = torch.mean(autocorrelations, dim=(1, -1))
        if self.training:
            autocorrelations_mean_on_bsz = torch.mean(autocorrelations_mean_on_head_channel, dim=0)
            _, top_k_delays_index = torch.topk(autocorrelations_mean_on_bsz, top_k)
            top_k_autocorrelations = torch.stack([autocorrelations_mean_on_head_channel[:, top_k_delays_index[i]] for i in range(top_k)], dim=-1)
        else:
            top_k_autocorrelations, top_k_delays_index = torch.topk(autocorrelations_mean_on_head_channel, top_k, dim=1)
        top_k_autocorrelations = torch.softmax(top_k_autocorrelations, dim=-1)
        if not self.training:
            tmp_values = value_states.repeat(1, 2, 1)
            init_index = torch.arange(time_length).view(1, -1, 1).repeat(bsz * self.num_heads, 1, channel).to(value_states.device)
        delays_agg = torch.zeros_like(value_states).float()
        for i in range(top_k):
            if not self.training:
                tmp_delay = init_index + top_k_delays_index[:, i].view(-1, 1, 1).repeat(self.num_heads, tgt_len, channel)
                value_states_roll_delay = torch.gather(tmp_values, dim=1, index=tmp_delay)
            else:
                value_states_roll_delay = value_states.roll(shifts=-int(top_k_delays_index[i]), dims=1)
            top_k_autocorrelations_at_delay = top_k_autocorrelations[:, i].view(-1, 1, 1).repeat(self.num_heads, tgt_len, channel)
            delays_agg += value_states_roll_delay * top_k_autocorrelations_at_delay
        attn_output = delays_agg.contiguous()
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped, past_key_value)