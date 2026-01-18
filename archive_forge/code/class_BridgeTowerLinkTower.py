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
class BridgeTowerLinkTower(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.link_tower_type = config.link_tower_type
        self.hidden_size = config.hidden_size
        if config.link_tower_type in ['add', 'scaled_add', 'interpolate']:
            if config.link_tower_type == 'scaled_add':
                self.scaled_factor = nn.Parameter(torch.tensor(1.0))
            elif config.link_tower_type == 'interpolate':
                self.beta = nn.Parameter(torch.tensor(0.5))
            self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        else:
            raise NotImplementedError(f'link_tower_type {config.link_tower_type} is not implemented')

    def forward(self, hidden_states, cross_modal_hidden_states, attention_mask):
        if self.link_tower_type == 'add':
            return self.LayerNorm(hidden_states + cross_modal_hidden_states)
        elif self.link_tower_type == 'scaled_add':
            return self.LayerNorm(hidden_states * self.scaled_factor + cross_modal_hidden_states)
        elif self.link_tower_type == 'interpolate':
            return self.LayerNorm(hidden_states * (1 - self.beta) + cross_modal_hidden_states * self.beta)
        else:
            raise NotImplementedError(f'link_tower_type {self.link_tower_type} is not implemented')