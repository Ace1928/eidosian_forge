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
class VanSpatialAttentionLayer(nn.Module):
    """
    Van spatial attention layer composed by projection (via conv) -> act -> Large Kernel Attention (LKA) attention ->
    projection (via conv) + residual connection.
    """

    def __init__(self, hidden_size: int, hidden_act: str='gelu'):
        super().__init__()
        self.pre_projection = nn.Sequential(OrderedDict([('conv', nn.Conv2d(hidden_size, hidden_size, kernel_size=1)), ('act', ACT2FN[hidden_act])]))
        self.attention_layer = VanLargeKernelAttentionLayer(hidden_size)
        self.post_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.pre_projection(hidden_state)
        hidden_state = self.attention_layer(hidden_state)
        hidden_state = self.post_projection(hidden_state)
        hidden_state = hidden_state + residual
        return hidden_state