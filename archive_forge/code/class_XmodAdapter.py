import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xmod import XmodConfig
class XmodAdapter(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bottleneck_size = config.hidden_size // config.adapter_reduction_factor
        self.dense1 = nn.Linear(config.hidden_size, self.bottleneck_size)
        self.dense2 = nn.Linear(self.bottleneck_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.adapter_act_fn = ACT2FN[config.hidden_act]
        else:
            self.adapter_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.adapter_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states