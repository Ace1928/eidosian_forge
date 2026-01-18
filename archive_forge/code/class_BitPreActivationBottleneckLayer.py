import collections
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_bit import BitConfig
class BitPreActivationBottleneckLayer(nn.Module):
    """Pre-activation (v2) bottleneck block.
    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, config, in_channels, out_channels=None, bottle_ratio=0.25, stride=1, dilation=1, first_dilation=None, groups=1, drop_path_rate=0.0, is_first_layer=False):
        super().__init__()
        first_dilation = first_dilation or dilation
        out_channels = out_channels or in_channels
        mid_channels = make_div(out_channels * bottle_ratio)
        if is_first_layer:
            self.downsample = BitDownsampleConv(config, in_channels, out_channels, stride=stride, preact=True)
        else:
            self.downsample = None
        self.norm1 = BitGroupNormActivation(config, in_channels)
        self.conv1 = WeightStandardizedConv2d(in_channels, mid_channels, 1, eps=1e-08, padding=config.global_padding)
        self.norm2 = BitGroupNormActivation(config, num_channels=mid_channels)
        self.conv2 = WeightStandardizedConv2d(mid_channels, mid_channels, 3, stride=stride, groups=groups, eps=1e-08, padding=config.global_padding)
        self.norm3 = BitGroupNormActivation(config, mid_channels)
        self.conv3 = WeightStandardizedConv2d(mid_channels, out_channels, 1, eps=1e-08, padding=config.global_padding)
        self.drop_path = BitDropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, hidden_states):
        hidden_states_preact = self.norm1(hidden_states)
        shortcut = hidden_states
        if self.downsample is not None:
            shortcut = self.downsample(hidden_states_preact)
        hidden_states = self.conv1(hidden_states_preact)
        hidden_states = self.conv2(self.norm2(hidden_states))
        hidden_states = self.conv3(self.norm3(hidden_states))
        hidden_states = self.drop_path(hidden_states)
        return hidden_states + shortcut