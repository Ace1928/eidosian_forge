import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_wavlm import WavLMConfig
class WavLMEncoderLayerStableLayerNorm(nn.Module):

    def __init__(self, config: WavLMConfig, has_relative_position_bias: bool=True):
        super().__init__()
        self.attention = WavLMAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.attention_dropout, num_buckets=config.num_buckets, max_distance=config.max_bucket_distance, has_relative_position_bias=has_relative_position_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, position_bias = self.attention(hidden_states, attention_mask=attention_mask, position_bias=position_bias, output_attentions=output_attentions)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
        outputs = (hidden_states, position_bias)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs