import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent, Identity
from .... import base
def _make_dense_block(num_layers, bn_size, growth_rate, dropout, stage_index):
    out = nn.HybridSequential(prefix='stage%d_' % stage_index)
    with out.name_scope():
        for _ in range(num_layers):
            out.add(_make_dense_layer(growth_rate, bn_size, dropout))
    return out