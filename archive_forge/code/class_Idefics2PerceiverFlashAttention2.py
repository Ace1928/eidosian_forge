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
class Idefics2PerceiverFlashAttention2(Idefics2PerceiverAttention):
    """
    Idefics2 flash attention module. This module inherits from `Idefics2PerceiverAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(self, latents: torch.Tensor, context: torch.Tensor, attention_mask: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Cache]=None, output_attentions: bool=False, use_cache: bool=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = latents.size()
        kv_seq_len = q_len + context.size()[1]
        query_states = self.q_proj(latents)
        key_states = self.k_proj(torch.cat([context, latents], dim=-2))
        value_states = self.v_proj(torch.cat([context, latents], dim=-2))
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        if past_key_value is not None:
            if hasattr(self.config, 'sliding_window') and kv_seq_len > self.config.sliding_window:
                slicing_tokens = kv_seq_len - self.config.sliding_window
                past_key = past_key_value[0]
                past_value = past_key_value[1]
                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()
                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(f'past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got {past_key.shape}')
                past_key_value = (past_key, past_value)
                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, '_pre_quantization_dtype'):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(f'The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}.')
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate, use_sliding_windows=False)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return (attn_output, attn_weights, past_key_value)

    def _flash_attention_forward(self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None, use_sliding_windows=False):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(query_states, key_states, value_states, attention_mask, query_length)
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k, dropout_p=dropout, softmax_scale=softmax_scale, causal=causal)
            else:
                attn_output_unpad = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k, dropout_p=dropout, softmax_scale=softmax_scale, causal=causal, window_size=(self.config.sliding_window, self.config.sliding_window))
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        elif not use_sliding_windows:
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal)
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal, window_size=(self.config.sliding_window, self.config.sliding_window))
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len:]
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        if query_length == kv_seq_len:
            query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
        return (query_layer, key_layer, value_layer, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k))