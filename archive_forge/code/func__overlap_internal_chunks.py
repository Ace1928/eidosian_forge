from __future__ import annotations
import warnings
from numbers import Integral, Number
import numpy as np
from tlz import concat, get, partial
from tlz.curried import map
from dask.array import chunk
from dask.array.core import Array, concatenate, map_blocks, unify_chunks
from dask.array.creation import empty_like, full_like
from dask.array.numpy_compat import normalize_axis_tuple
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayOverlapLayer
from dask.utils import derived_from
def _overlap_internal_chunks(original_chunks, axes):
    """Get new chunks for array with overlap."""
    chunks = []
    for i, bds in enumerate(original_chunks):
        depth = axes.get(i, 0)
        if isinstance(depth, tuple):
            left_depth = depth[0]
            right_depth = depth[1]
        else:
            left_depth = depth
            right_depth = depth
        if len(bds) == 1:
            chunks.append(bds)
        else:
            left = [bds[0] + right_depth]
            right = [bds[-1] + left_depth]
            mid = []
            for bd in bds[1:-1]:
                mid.append(bd + left_depth + right_depth)
            chunks.append(left + mid + right)
    return chunks