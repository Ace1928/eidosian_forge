import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_xlm_prophetnet import XLMProphetNetConfig
class XLMProphetNetEncoderLayer(nn.Module):
    """
    Encoder block for XLMProphetnet
    """

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__()
        self.self_attn = XLMProphetNetAttention(config, config.num_encoder_attention_heads)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)
        self.feed_forward = XLMProphetNetFeedForward(config, config.encoder_ffn_dim)
        self.feed_forward_layer_norm = LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions: bool=False):
        attention_output, attn_weights, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = self.self_attn_layer_norm(attention_output + hidden_states)
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs