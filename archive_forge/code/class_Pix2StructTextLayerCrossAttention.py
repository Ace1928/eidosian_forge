import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_pix2struct import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig
class Pix2StructTextLayerCrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = Pix2StructTextAttention(config, has_relative_attention_bias=False)
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, key_value_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, query_length=None, output_attentions=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.attention(normed_hidden_states, mask=attention_mask, key_value_states=key_value_states, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, query_length=query_length, output_attentions=output_attentions)
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs