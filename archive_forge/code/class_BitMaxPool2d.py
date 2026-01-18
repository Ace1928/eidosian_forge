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
class BitMaxPool2d(nn.MaxPool2d):
    """Tensorflow like 'SAME' wrapper for 2D max pooling"""

    def __init__(self, kernel_size: int, stride=None, dilation=1, ceil_mode=False, padding=(0, 0), padding_value=0, use_dynamic_padding=True):
        kernel_size = kernel_size if isinstance(kernel_size, collections.abc.Iterable) else (kernel_size, kernel_size)
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        dilation = dilation if isinstance(dilation, collections.abc.Iterable) else (dilation, dilation)
        super().__init__(kernel_size, stride, padding, dilation, ceil_mode)
        if use_dynamic_padding:
            self.pad = DynamicPad2d(kernel_size, stride, dilation, padding_value)
        else:
            self.pad = nn.Identity()

    def forward(self, hidden_states):
        hidden_states = self.pad(hidden_states)
        return nn.functional.max_pool2d(hidden_states, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)