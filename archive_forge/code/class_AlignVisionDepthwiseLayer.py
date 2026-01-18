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
class AlignVisionDepthwiseLayer(nn.Module):
    """
    This corresponds to the depthwise convolution phase of each block in the original implementation.
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, stride: int, kernel_size: int, adjust_padding: bool):
        super().__init__()
        self.stride = stride
        conv_pad = 'valid' if self.stride == 2 else 'same'
        padding = correct_pad(kernel_size, adjust=adjust_padding)
        self.depthwise_conv_pad = nn.ZeroPad2d(padding=padding)
        self.depthwise_conv = AlignVisionDepthwiseConv2d(in_dim, kernel_size=kernel_size, stride=stride, padding=conv_pad, bias=False)
        self.depthwise_norm = nn.BatchNorm2d(num_features=in_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        self.depthwise_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        if self.stride == 2:
            hidden_states = self.depthwise_conv_pad(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_norm(hidden_states)
        hidden_states = self.depthwise_act(hidden_states)
        return hidden_states