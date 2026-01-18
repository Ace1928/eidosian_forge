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
class SeamlessM4Tv2ConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.speech_encoder_dropout
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = SeamlessM4Tv2ConformerFeedForward(config)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config)
        self.conv_module = SeamlessM4Tv2ConformerConvolutionModule(config)
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = SeamlessM4Tv2ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, conv_attention_mask: Optional[torch.Tensor]=None):
        hidden_states = hidden_states
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)
        return (hidden_states, attn_weights)