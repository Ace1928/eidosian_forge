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
class LongT5LayerFF(nn.Module):

    def __init__(self, config: LongT5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = LongT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = LongT5DenseActDense(config)
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states