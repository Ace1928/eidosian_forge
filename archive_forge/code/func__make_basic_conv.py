import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent
from .... import base
def _make_basic_conv(**kwargs):
    out = nn.HybridSequential(prefix='')
    out.add(nn.Conv2D(use_bias=False, **kwargs))
    out.add(nn.BatchNorm(epsilon=0.001))
    out.add(nn.Activation('relu'))
    return out