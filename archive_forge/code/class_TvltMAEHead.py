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
class TvltMAEHead(nn.Module):

    def __init__(self, config, output_dim=None):
        super().__init__()
        self.config = config
        self.decoder = nn.Linear(config.decoder_hidden_size, output_dim)

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states