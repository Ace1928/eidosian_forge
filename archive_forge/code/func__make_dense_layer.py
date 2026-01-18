import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent, Identity
from .... import base
def _make_dense_layer(growth_rate, bn_size, dropout):
    new_features = nn.HybridSequential(prefix='')
    new_features.add(nn.BatchNorm())
    new_features.add(nn.Activation('relu'))
    new_features.add(nn.Conv2D(bn_size * growth_rate, kernel_size=1, use_bias=False))
    new_features.add(nn.BatchNorm())
    new_features.add(nn.Activation('relu'))
    new_features.add(nn.Conv2D(growth_rate, kernel_size=3, padding=1, use_bias=False))
    if dropout:
        new_features.add(nn.Dropout(dropout))
    out = HybridConcurrent(axis=1, prefix='')
    out.add(Identity())
    out.add(new_features)
    return out