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
class VanLargeKernelAttention(nn.Module):
    """
    Basic Large Kernel Attention (LKA).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.depth_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size)
        self.depth_wise_dilated = nn.Conv2d(hidden_size, hidden_size, kernel_size=7, dilation=3, padding=9, groups=hidden_size)
        self.point_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.depth_wise(hidden_state)
        hidden_state = self.depth_wise_dilated(hidden_state)
        hidden_state = self.point_wise(hidden_state)
        return hidden_state