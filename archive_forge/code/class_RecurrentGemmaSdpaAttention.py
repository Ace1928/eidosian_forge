import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_recurrent_gemma import RecurrentGemmaConfig
class RecurrentGemmaSdpaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: RecurrentGemmaConfig):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.partial_rotary_factor = config.partial_rotary_factor
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=True)
        self.rotary_emb = RecurrentGemmaRotaryEmbedding(int(self.partial_rotary_factor * self.head_dim), base=config.rope_theta)

    def forward(self, hidden_states: torch.Tensor, position_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, cache_position: Optional[torch.LongTensor]=None, use_cache: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_rot, query_pass = torch.chunk(query_states, int(1 / self.partial_rotary_factor), dim=-1)
        key_rot, key_pass = torch.chunk(key_states, int(1 / self.partial_rotary_factor), dim=-1)
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)
        if use_cache and hasattr(self, 'key_states'):
            cache_kwargs = {'cache_position': cache_position}
            key_states, value_states = self._update_cache(key_states, value_states, **cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]
        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states.contiguous(), key_states.contiguous(), value_states.contiguous(), attn_mask=causal_mask, dropout_p=self.attention_dropout if self.training else 0.0, scale=self.head_dim ** (-0.5))
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _setup_cache(self, batch_size, device, dtype=None):
        if dtype is None and self.config.torch_dtype is not None:
            dtype = self.config.torch_dtype
        dtype = dtype if dtype is not None else torch.float32
        cache_shape = (batch_size, self.num_key_value_heads, self.config.attention_window_size, self.head_dim)
        self.value_states = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.key_states = torch.zeros(cache_shape, dtype=dtype, device=device)

    @torch.no_grad()
    def _update_cache(self, key_states, value_states, **cache_kwargs):
        """
        torch.compile compatible sliding window.
        Computes the `indices` based on `cache_position >= self.config.attention_window_size - 1`.
        The `to_shift` is only true once we are above attention_window_size. Thus with `attention_window_size==64`:

        indices = (slicing + to_shift[-1].int()-1) % self.config.attention_window_size
        tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63,  0])

        We overwrite the cache using these, then we always write at cache_position (clamped to `attention_window_size`)
        """
        cache_position = cache_kwargs.get('cache_position')
        if cache_position.shape[0] > self.config.attention_window_size:
            k_out = key_states[:, :, -self.config.attention_window_size:, :]
            v_out = value_states[:, :, -self.config.attention_window_size:, :]
        else:
            slicing = torch.ones(self.config.attention_window_size, dtype=torch.long, device=value_states.device).cumsum(0)
            cache_position = cache_position.clamp(0, self.config.attention_window_size - 1)
            to_shift = cache_position >= self.config.attention_window_size - 1
            indices = (slicing + to_shift[-1].int() - 1) % self.config.attention_window_size
            k_out, v_out = (self.key_states.to(key_states.device), self.value_states.to(value_states.device))
            k_out = k_out[:, :, indices]
            v_out = v_out[:, :, indices]
            k_out[:, :, cache_position] = key_states
            v_out[:, :, cache_position] = value_states
        self.key_states, self.value_states = (k_out, v_out)
        return (k_out, v_out)