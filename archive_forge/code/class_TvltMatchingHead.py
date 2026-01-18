import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
class TvltMatchingHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.pooler = TvltPooler(config)
        self.fc = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        hidden_states = self.fc(self.pooler(hidden_states))
        return hidden_states