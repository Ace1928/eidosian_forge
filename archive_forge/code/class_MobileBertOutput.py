import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
class MobileBertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.dense = nn.Linear(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size)
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        else:
            self.bottleneck = OutputBottleneck(config)

    def forward(self, intermediate_states: torch.Tensor, residual_tensor_1: torch.Tensor, residual_tensor_2: torch.Tensor) -> torch.Tensor:
        layer_output = self.dense(intermediate_states)
        if not self.use_bottleneck:
            layer_output = self.dropout(layer_output)
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        else:
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
            layer_output = self.bottleneck(layer_output, residual_tensor_2)
        return layer_output