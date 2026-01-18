import math
from collections import OrderedDict
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....activations import ACT2FN
from ....modeling_outputs import (
from ....modeling_utils import PreTrainedModel
from ....utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_van import VanConfig
class VanLayerScaling(nn.Module):
    """
    Scales the inputs by a learnable parameter initialized by `initial_value`.
    """

    def __init__(self, hidden_size: int, initial_value: float=0.01):
        super().__init__()
        self.weight = nn.Parameter(initial_value * torch.ones(hidden_size), requires_grad=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.weight.unsqueeze(-1).unsqueeze(-1) * hidden_state
        return hidden_state