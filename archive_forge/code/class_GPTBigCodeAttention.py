import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_gpt_bigcode import GPTBigCodeConfig
class GPTBigCodeAttention(nn.Module):

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        self.mask_value = None
        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_heads = 1 if self.multi_query else self.num_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.split_size = self.embed_dim
        self.is_causal = True
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        self.attn_pdrop = config.attn_pdrop
        if self.is_cross_attention:
            if self.multi_query:
                raise NotImplementedError('Multi-Query Attention not supported for cross_attention')
            self.c_attn = nn.Linear(self.embed_dim, 2 * self.embed_dim)
            self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = nn.Linear(self.embed_dim, self.embed_dim + 2 * self.kv_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _get_mask_value(self, device, dtype):
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        return self.mask_value

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype
        upcast = dtype != softmax_dtype
        unscale = self.layer_idx + 1 if self.scale_attention_softmax_in_fp32 and upcast else 1
        scale_factor = unscale ** (-1)
        if self.scale_attn_weights:
            scale_factor /= self.head_dim ** 0.5
        query_shape = query.shape
        batch_size = query_shape[0]
        key_length = key.size(-1)
        if self.multi_query:
            query_length = query_shape[1]
            attn_shape = (batch_size, query_length, self.num_heads, key_length)
            attn_view = (batch_size, query_length * self.num_heads, key_length)
            query = query.reshape(batch_size, query_length * self.num_heads, self.head_dim)
        else:
            query_length = query_shape[2]
            attn_shape = (batch_size, self.num_heads, query_length, key_length)
            attn_view = (batch_size * self.num_heads, query_length, key_length)
            query = query.reshape(batch_size * self.num_heads, query_length, self.head_dim)
            key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)
        attn_weights = torch.empty(attn_view, device=query.device, dtype=query.dtype)
        if query.device.type == 'cpu':
            attn_weights = torch.zeros_like(attn_weights)
            beta = 1
        else:
            beta = 0
        attn_weights = torch.baddbmm(attn_weights, query, key, beta=beta, alpha=scale_factor).view(attn_shape)
        if upcast:
            if attention_mask is None:
                attn_weights = upcast_softmax(attn_weights, unscale, softmax_dtype)
            else:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)
                attn_weights = upcast_masked_softmax(attn_weights, attention_mask, mask_value, unscale, softmax_dtype)
        else:
            if attention_mask is not None:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)
                attn_weights = torch.where(attention_mask, attn_weights, mask_value)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            if self.multi_query:
                head_mask = head_mask.transpose(1, 2)
            attn_weights = attn_weights * head_mask
        if self.multi_query:
            attn_output = torch.bmm(attn_weights.view(attn_view), value).view(query_shape)
        else:
            attn_output = torch.matmul(attn_weights, value)
        return (attn_output, attn_weights)

    def forward(self, hidden_states: torch.Tensor, layer_past: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]]]:
        if encoder_hidden_states is not None:
            if not hasattr(self, 'q_attn') or not self.is_cross_attention:
                raise ValueError('If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`.')
            query = self.q_attn(hidden_states)
            key_value = self.c_attn(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        elif self.multi_query:
            query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
        else:
            query, key_value = self.c_attn(hidden_states).view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim).transpose(1, 2).split((self.head_dim, 2 * self.head_dim), dim=3)
        if layer_past is not None:
            key_value = torch.cat((layer_past, key_value), dim=-2)
        present = key_value if use_cache else None
        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)
        attn_output, attn_weights = self._attn(query, key.transpose(-1, -2), value, attention_mask, head_mask)
        if not self.multi_query:
            attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            if self.multi_query:
                attn_weights = attn_weights.transpose(1, 2)
            outputs += (attn_weights,)
        return outputs