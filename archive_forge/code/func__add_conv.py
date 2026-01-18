import os
from ... import nn
from ....context import cpu
from ...block import HybridBlock
from .... import base
def _add_conv(out, channels=1, kernel=1, stride=1, pad=0, num_group=1, active=True, relu6=False):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True))
    if active:
        out.add(RELU6() if relu6 else nn.Activation('relu'))