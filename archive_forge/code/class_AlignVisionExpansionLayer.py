import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_align import AlignConfig, AlignTextConfig, AlignVisionConfig
class AlignVisionExpansionLayer(nn.Module):
    """
    This corresponds to the expansion phase of each block in the original implementation.
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int):
        super().__init__()
        self.expand_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, padding='same', bias=False)
        self.expand_bn = nn.BatchNorm2d(num_features=out_dim, eps=config.batch_norm_eps)
        self.expand_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.expand_conv(hidden_states)
        hidden_states = self.expand_bn(hidden_states)
        hidden_states = self.expand_act(hidden_states)
        return hidden_states