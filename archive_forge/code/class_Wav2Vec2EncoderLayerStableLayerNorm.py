import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
from .configuration_wav2vec2 import Wav2Vec2Config
class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.attention_dropout, is_decoder=False)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if getattr(config, 'adapter_attn_dim', None) is not None:
            self.adapter_layer = Wav2Vec2AttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs