import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent
from .... import base
def _make_E(prefix):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, (320, 1, None, None)))
        branch_3x3 = nn.HybridSequential(prefix='')
        out.add(branch_3x3)
        branch_3x3.add(_make_branch(None, (384, 1, None, None)))
        branch_3x3_split = HybridConcurrent(axis=1, prefix='')
        branch_3x3_split.add(_make_branch(None, (384, (1, 3), None, (0, 1))))
        branch_3x3_split.add(_make_branch(None, (384, (3, 1), None, (1, 0))))
        branch_3x3.add(branch_3x3_split)
        branch_3x3dbl = nn.HybridSequential(prefix='')
        out.add(branch_3x3dbl)
        branch_3x3dbl.add(_make_branch(None, (448, 1, None, None), (384, 3, None, 1)))
        branch_3x3dbl_split = HybridConcurrent(axis=1, prefix='')
        branch_3x3dbl.add(branch_3x3dbl_split)
        branch_3x3dbl_split.add(_make_branch(None, (384, (1, 3), None, (0, 1))))
        branch_3x3dbl_split.add(_make_branch(None, (384, (3, 1), None, (1, 0))))
        out.add(_make_branch('avg', (192, 1, None, None)))
    return out