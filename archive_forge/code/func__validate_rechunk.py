from __future__ import annotations
import heapq
import math
from functools import reduce
from itertools import chain, count, product
from operator import add, itemgetter, mul
from warnings import warn
import numpy as np
import tlz as toolz
from tlz import accumulate
from dask import config
from dask.array.chunk import getitem
from dask.array.core import Array, concatenate3, normalize_chunks
from dask.array.utils import validate_axis
from dask.array.wrap import empty
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import parse_bytes
def _validate_rechunk(old_chunks, new_chunks):
    """Validates that rechunking an array from ``old_chunks`` to ``new_chunks``
    is possible, raises an error if otherwise.

    Notes
    -----
    This function expects ``old_chunks`` and ``new_chunks`` to have matching
    dimensionality and will not raise an informative error if they don't.
    """
    assert len(old_chunks) == len(new_chunks)
    old_shapes = tuple(map(sum, old_chunks))
    new_shapes = tuple(map(sum, new_chunks))
    for old_shape, old_dim, new_shape, new_dim in zip(old_shapes, old_chunks, new_shapes, new_chunks):
        if old_shape != new_shape:
            if not (math.isnan(old_shape) and math.isnan(new_shape)) or not np.array_equal(old_dim, new_dim, equal_nan=True):
                raise ValueError('Chunks must be unchanging along dimensions with missing values.\n\nA possible solution:\n  x.compute_chunk_sizes()')