import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent
from .... import base
def _make_D(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, (192, 1, None, None), (320, 3, 2, None)))
        out.add(_make_branch(None, (192, 1, None, None), (192, (1, 7), None, (0, 3)), (192, (7, 1), None, (3, 0)), (192, 3, 2, None)))
        out.add(_make_branch('max'))
    return out