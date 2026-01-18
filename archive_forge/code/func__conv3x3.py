import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from .... import base
from .... util import is_np_array
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1, use_bias=False, in_channels=in_channels)