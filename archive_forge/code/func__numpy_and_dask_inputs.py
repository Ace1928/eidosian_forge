from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
def _numpy_and_dask_inputs(input_sigs):
    _dimensions = {'a': 5, 'b': 6, 'c': 7, 'd': 5, 'e': 6, 'f': 10, 'g': 1, 'h': 2, '*': 11}
    _chunks = {'a': (2, 3), 'b': (2, 3, 1), 'c': (2, 3, 2), 'd': (4, 1), 'e': (2, 4), 'f': (1, 2, 3, 4), 'g': 1, 'h': (1, 1), '*': 11}

    def _shape_from_string(s):
        return tuple((_dimensions[c] for c in s))

    def _chunks_from_string(s):
        return tuple((_chunks[c] for c in s))
    shapes = [_shape_from_string(s) for s in input_sigs]
    chunks = [_chunks_from_string(s) for s in input_sigs]
    np_inputs = [np.random.default_rng().random(s) for s in shapes]
    da_inputs = [da.from_array(i, chunks=c) for i, c in zip(np_inputs, chunks)]
    return (np_inputs, da_inputs)