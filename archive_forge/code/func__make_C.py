import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent
from .... import base
def _make_C(channels_7x7, prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, (192, 1, None, None)))
        out.add(_make_branch(None, (channels_7x7, 1, None, None), (channels_7x7, (1, 7), None, (0, 3)), (192, (7, 1), None, (3, 0))))
        out.add(_make_branch(None, (channels_7x7, 1, None, None), (channels_7x7, (7, 1), None, (3, 0)), (channels_7x7, (1, 7), None, (0, 3)), (channels_7x7, (7, 1), None, (3, 0)), (192, (1, 7), None, (0, 3))))
        out.add(_make_branch('avg', (192, 1, None, None)))
    return out