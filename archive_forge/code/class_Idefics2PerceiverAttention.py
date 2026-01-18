import inspect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...utils import (
from ..auto import AutoModel
from .configuration_idefics2 import Idefics2Config, Idefics2VisionConfig
class Idefics2PerceiverAttention(nn.Module):

    def __init__(self, config, layer_idx: Optional[int]=None) -> None:
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        super().__init__()
        self.layer_idx = None
        self.hidden_size = config.text_config.hidden_size
        self.num_heads = config.perceiver_config.resampler_n_heads
        self.head_dim = config.perceiver_config.resampler_head_dim
        self.num_key_value_heads = config.perceiver_config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.perceiver_config.attention_dropout
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.is_causal = False

    def forward(self, latents: torch.Tensor, context: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: bool=False, use_cache: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!

        Args:
            latents (`torch.Tensor`): Tensor of shape [bsz, n_latents, embed_dim] representing fixed length latents to compress to.
            context (`torch.Tensor`): Tensor of shape [bsz, seq, embed_dim] representing long-form context to resample.
            attention_mask (`torch.Tensor`, *optional*): Tensor of shape [bsz, 1, seq, n_latents] representing attention mask.
            position_ids (`torch.LongTensor`, *optional*): Tensor of shape [bsz, seq] representing position indices of each input token.
            past_key_value (`Tuple[torch.Tensor]`, *optional*): Tuple of tensors containing cached key and value states.
            output_attentions (`bool`, *optional*, defaults to `False`): Whether to return attention weights.
            use_cache (`bool`, *optional*, defaults to `False`): Whether to use past_key_value for caching.
        """
        bsz, q_len, _ = latents.size()
        kv_seq_len = q_len + context.size()[1]
        hidden_states = torch.concat([context, latents], dim=-2)
        query_states = self.q_proj(latents)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        past_key_value = getattr(self, 'past_key_value', past_key_value)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return (attn_output, attn_weights, past_key_value)