from __future__ import annotations
import contextlib
import functools
import itertools
import math
import numbers
import warnings
import numpy as np
from tlz import concat, frequencies
from dask.array.core import Array
from dask.array.numpy_compat import AxisError
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import has_keyword, is_arraylike, is_cupy_type, typename
def _check_chunks(x, check_ndim=True, scheduler=None):
    x = x.persist(scheduler=scheduler)
    for idx in itertools.product(*(range(len(c)) for c in x.chunks)):
        chunk = x.dask[(x.name,) + idx]
        if hasattr(chunk, 'result'):
            chunk = chunk.result()
        if not hasattr(chunk, 'dtype'):
            chunk = np.array(chunk, dtype='O')
        expected_shape = tuple((c[i] for c, i in zip(x.chunks, idx)))
        assert_eq_shape(expected_shape, chunk.shape, check_ndim=check_ndim, check_nan=False)
        assert chunk.dtype == x.dtype, 'maybe you forgot to pass the scheduler to `assert_eq`?'
    return x