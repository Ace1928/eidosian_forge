import copy
import math
import warnings
from typing import Any, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_longt5 import LongT5Config
class LongT5LayerLocalSelfAttention(nn.Module):
    """Local self attention used in encoder"""

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.LocalSelfAttention = LongT5LocalAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, layer_head_mask=None, output_attentions=False, **kwargs: Any):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.LocalSelfAttention(normed_hidden_states, mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs