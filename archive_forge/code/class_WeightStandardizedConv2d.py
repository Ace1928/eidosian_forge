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
class WeightStandardizedConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Includes TensorFlow compatible SAME padding. Used for ViT Hybrid model.

    Paper: [Micro-Batch Training with Batch-Channel Normalization and Weight
    Standardization](https://arxiv.org/abs/1903.10520v2)
    """

    def __init__(self, in_channel, out_channels, kernel_size, stride=1, padding='SAME', dilation=1, groups=1, bias=False, eps=1e-06):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(in_channel, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if is_dynamic:
            self.pad = DynamicPad2d(kernel_size, stride, dilation)
        else:
            self.pad = None
        self.eps = eps

    def forward(self, hidden_state):
        if self.pad is not None:
            hidden_state = self.pad(hidden_state)
        weight = nn.functional.batch_norm(self.weight.reshape(1, self.out_channels, -1), None, None, training=True, momentum=0.0, eps=self.eps).reshape_as(self.weight)
        hidden_state = nn.functional.conv2d(hidden_state, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return hidden_state