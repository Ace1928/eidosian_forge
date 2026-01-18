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
class MobileBertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = MobileBertPredictionHeadTransform(config)
        self.dense = nn.Linear(config.vocab_size, config.hidden_size - config.embedding_size, bias=False)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        hidden_states += self.decoder.bias
        return hidden_states