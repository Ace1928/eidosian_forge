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
class DynamicPad2d(nn.Module):
    """
    A module that wraps dynamic padding of any input, given the parameters of the convolutional layer and the input
    hidden states.
    """

    def __init__(self, kernel_size, stride, dilation, value=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.value = value

        def compute_padding(x, kernel_size, stride, dilation):
            return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)
        self.compute_padding = compute_padding

    def __call__(self, input):
        input_height, input_width = input.size()[-2:]
        padding_height = self.compute_padding(input_height, self.kernel_size[0], self.stride[0], self.dilation[0])
        padding_width = self.compute_padding(input_width, self.kernel_size[1], self.stride[1], self.dilation[1])
        if padding_height > 0 or padding_width > 0:
            input = nn.functional.pad(input, [padding_width // 2, padding_width - padding_width // 2, padding_height // 2, padding_height - padding_height // 2], value=self.value)
        return input