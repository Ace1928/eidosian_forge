import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import (
from .configuration_decision_transformer import DecisionTransformerConfig
def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
    bsz, num_heads, q_seq_len, dk = query.size()
    _, _, k_seq_len, _ = key.size()
    attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)
    scale_factor = 1.0
    if self.scale_attn_weights:
        scale_factor /= float(value.size(-1)) ** 0.5
    if self.scale_attn_by_inverse_layer_idx:
        scale_factor /= float(self.layer_idx + 1)
    with autocast(enabled=False):
        q, k = (query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len))
        attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
        attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)
    if not self.is_cross_attention:
        query_length, key_length = (query.size(-2), key.size(-2))
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    if attn_weights.dtype != torch.float32:
        raise RuntimeError('Error with upcasting, attn_weights does not have dtype torch.float32')
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_output = torch.matmul(attn_weights, value)
    return (attn_output, attn_weights)