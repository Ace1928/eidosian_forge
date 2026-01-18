import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_encodec import EncodecConfig
class EncodecConvTranspose1d(nn.Module):
    """ConvTranspose1d with asymmetric or causal padding and normalization."""

    def __init__(self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int=1):
        super().__init__()
        self.causal = config.use_causal_conv
        self.trim_right_ratio = config.trim_right_ratio
        self.norm_type = config.norm_type
        if self.norm_type not in ['weight_norm', 'time_group_norm']:
            raise ValueError(f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}')
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        if config.norm_type == 'weight_norm':
            self.conv = nn.utils.weight_norm(self.conv)
        elif config.norm_type == 'time_group_norm':
            self.norm = nn.GroupNorm(1, out_channels)
        if not (self.causal or self.trim_right_ratio == 1.0):
            raise ValueError('`trim_right_ratio` != 1.0 only makes sense for causal convolutions')

    def forward(self, hidden_states):
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride
        hidden_states = self.conv(hidden_states)
        if self.norm_type == 'time_group_norm':
            hidden_states = self.norm(hidden_states)
        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
        else:
            padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        end = hidden_states.shape[-1] - padding_right
        hidden_states = hidden_states[..., padding_left:end]
        return hidden_states