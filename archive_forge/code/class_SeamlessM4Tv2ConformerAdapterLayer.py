import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
class SeamlessM4Tv2ConformerAdapterLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.adaptor_dropout
        self.kernel_size = config.adaptor_kernel_size
        self.stride = config.adaptor_stride
        self.residual_layer_norm = nn.LayerNorm(embed_dim)
        self.residual_conv = nn.Conv1d(embed_dim, 2 * embed_dim, self.kernel_size, stride=self.stride, padding=self.stride // 2)
        self.activation = nn.GLU(dim=1)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_conv = nn.Conv1d(embed_dim, 2 * embed_dim, self.kernel_size, stride=self.stride, padding=self.stride // 2)
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config, use_position_embeddings=False)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = SeamlessM4Tv2ConformerFeedForward(config, act_fn='relu', dropout=dropout)

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        pad = self.kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)
        seq_lens = (seq_lens + 2 * pad - self.kernel_size) / self.stride + 1
        return seq_lens.floor()

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False):
        residual = self.residual_layer_norm(hidden_states)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        residual = residual.transpose(1, 2)
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(hidden_states.device)
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        hidden_states, attn_weigths = self.self_attn(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states) + residual
        return hidden_states