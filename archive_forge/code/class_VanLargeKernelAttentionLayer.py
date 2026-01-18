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
class VanLargeKernelAttentionLayer(nn.Module):
    """
    Computes attention using Large Kernel Attention (LKA) and attends the input.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = VanLargeKernelAttention(hidden_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        attention = self.attention(hidden_state)
        attended = hidden_state * attention
        return attended