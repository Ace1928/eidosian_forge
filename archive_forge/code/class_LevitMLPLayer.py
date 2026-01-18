import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_levit import LevitConfig
class LevitMLPLayer(nn.Module):
    """
    MLP Layer with `2X` expansion in contrast to ViT with `4X`.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_up = MLPLayerWithBN(input_dim, hidden_dim)
        self.activation = nn.Hardswish()
        self.linear_down = MLPLayerWithBN(hidden_dim, input_dim)

    def forward(self, hidden_state):
        hidden_state = self.linear_up(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.linear_down(hidden_state)
        return hidden_state