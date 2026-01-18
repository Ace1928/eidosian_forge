import os
from ... import nn
from ....context import cpu
from ...block import HybridBlock
from .... import base
def _add_conv_dw(out, dw_channels, channels, stride, relu6=False):
    _add_conv(out, channels=dw_channels, kernel=3, stride=stride, pad=1, num_group=dw_channels, relu6=relu6)
    _add_conv(out, channels=channels, relu6=relu6)