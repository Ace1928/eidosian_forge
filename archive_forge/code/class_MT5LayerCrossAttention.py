import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_mt5 import MT5Config
class MT5LayerCrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = MT5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, key_value_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, query_length=None, output_attentions=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(normed_hidden_states, mask=attention_mask, key_value_states=key_value_states, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, query_length=query_length, output_attentions=output_attentions)
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs