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
class XLMProphetNetFeedForward(nn.Module):
    """
    This is the residual two feed-forward layer block based on the original Transformer implementation.
    """

    def __init__(self, config: XLMProphetNetConfig, ffn_dim: int):
        super().__init__()
        self.activation_fn = ACT2FN[config.activation_function]
        self.intermediate = nn.Linear(config.hidden_size, ffn_dim)
        self.output = nn.Linear(ffn_dim, config.hidden_size)
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout

    def forward(self, hidden_states):
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.output(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states