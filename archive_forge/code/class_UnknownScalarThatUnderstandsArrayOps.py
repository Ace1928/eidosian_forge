from __future__ import annotations
import operator
import numpy as np
import pytest
import dask.array as da
from dask.array import Array
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.utils import assert_eq
class UnknownScalarThatUnderstandsArrayOps(np.lib.mixins.NDArrayOperatorsMixin):

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        outputs = kwargs.get('out', ())
        for item in inputs + outputs:
            if hasattr(item, '__array_ufunc__') and (not isinstance(item, (np.ndarray, Array, UnknownScalarThatUnderstandsArrayOps))):
                return NotImplemented
        return UnknownScalarThatUnderstandsArrayOps()