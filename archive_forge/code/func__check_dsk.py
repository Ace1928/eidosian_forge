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
def _check_dsk(dsk):
    """Check that graph is well named and non-overlapping"""
    if not isinstance(dsk, HighLevelGraph):
        return
    dsk.validate()
    assert all((isinstance(k, (tuple, str)) for k in dsk.layers))
    freqs = frequencies(concat(dsk.layers.values()))
    non_one = {k: v for k, v in freqs.items() if v != 1}
    key_collisions = set()
    for k in non_one.keys():
        for layer in dsk.layers.values():
            try:
                key_collisions.add(tokenize(layer[k]))
            except KeyError:
                pass
    assert len(key_collisions) < 2, non_one