import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig
class BridgeTowerTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        if config.remove_last_layer:
            self.resblocks = nn.ModuleList([BridgeTowerResidualAttention(config) for _ in range(self.num_hidden_layers - 1)])
        else:
            self.resblocks = nn.ModuleList([BridgeTowerResidualAttention(config) for _ in range(self.num_hidden_layers)])
        self.stop_gradient = config.stop_gradient

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]=None):
        hidden_states = []
        for block in self.resblocks:
            hidden_state = block(hidden_state, attention_mask)
            if self.stop_gradient:
                hidden_states.append(hidden_state.detach())
            else:
                hidden_states.append(hidden_state)
        return hidden_states