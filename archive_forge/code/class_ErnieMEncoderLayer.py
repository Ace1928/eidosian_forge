import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn, tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ernie_m import ErnieMConfig
class ErnieMEncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        dropout = 0.1 if config.hidden_dropout_prob is None else config.hidden_dropout_prob
        act_dropout = config.hidden_dropout_prob if config.act_dropout is None else config.act_dropout
        self.self_attn = ErnieMAttention(config)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=True):
        residual = hidden_states
        if output_attentions:
            hidden_states, attention_opt_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, head_mask=head_mask, past_key_value=past_key_value, output_attentions=output_attentions)
        else:
            hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, head_mask=head_mask, past_key_value=past_key_value, output_attentions=output_attentions)
        hidden_states = residual + self.dropout1(hidden_states)
        hidden_states = self.norm1(hidden_states)
        residual = hidden_states
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = residual + self.dropout2(hidden_states)
        hidden_states = self.norm2(hidden_states)
        if output_attentions:
            return (hidden_states, attention_opt_weights)
        else:
            return hidden_states