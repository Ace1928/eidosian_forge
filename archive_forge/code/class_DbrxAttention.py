import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_dbrx import DbrxConfig
class DbrxAttention(nn.Module):
    """Multi-head self attention."""

    def __init__(self, config: DbrxConfig, block_idx: Optional[int]=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_seq_len
        self.block_idx = block_idx
        if block_idx is None:
            logger.warning_once(f'Instantiating {self.__class__.__name__} without passing a `block_idx` is not recommended and will ' + 'lead to errors during the forward call if caching is used. Please make sure to provide a `block_idx` ' + 'when creating this class.')
        attn_config = config.attn_config
        self.attn_pdrop = attn_config.attn_pdrop
        self.clip_qkv = attn_config.clip_qkv
        self.num_key_value_heads = attn_config.kv_n_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = attn_config.rope_theta
        self.is_causal = True
        self.Wqkv = nn.Linear(self.hidden_size, self.hidden_size + 2 * self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = DbrxRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Cache]=None, output_attentions: bool=False, use_cache: bool=False, cache_position: Optional[torch.LongTensor]=None, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        bsz, q_len, _ = hidden_states.size()
        qkv_states = self.Wqkv(hidden_states)
        min_val = -self.clip_qkv if self.clip_qkv is not None else None
        max_val = self.clip_qkv
        qkv_states = qkv_states.clamp(min=min_val, max=max_val)
        query_states, key_states, value_states = qkv_states.split([self.hidden_size, self.num_key_value_heads * self.head_dim, self.num_key_value_heads * self.head_dim], dim=2)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        past_key_value = getattr(self, 'past_key_value', past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.block_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attn_pdrop, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is' + f' {attn_output.size()}')
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return (attn_output, attn_weights, past_key_value)