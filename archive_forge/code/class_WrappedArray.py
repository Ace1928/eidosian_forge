from __future__ import annotations
import operator
import numpy as np
import pytest
import dask.array as da
from dask.array import Array
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.utils import assert_eq
class WrappedArray(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Another mock duck array class (like EncapsulateNDArray), but
    designed to be above Dask in the type casting hierarchy (that is,
    WrappedArray wraps Dask Array) and be even more minimal in API.
    Tests that Dask defers properly to upcast types.
    """

    def __init__(self, arr, **attrs):
        self.arr = arr
        self.attrs = attrs

    def __array__(self, *args, **kwargs):
        return np.asarray(self.arr, *args, **kwargs)

    def _downcast_args(self, args):
        for arg in args:
            if isinstance(arg, type(self)):
                yield arg.arr
            elif isinstance(arg, (tuple, list)):
                yield type(arg)(self._downcast_args(arg))
            else:
                yield arg

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(self._downcast_args(inputs))
        return type(self)(getattr(ufunc, method)(*inputs, **kwargs), **self.attrs)

    def __array_function__(self, func, types, args, kwargs):
        args = tuple(self._downcast_args(args))
        return type(self)(func(*args, **kwargs), **self.attrs)

    def __dask_graph__(self):
        return ...
    shape = dispatch_property('shape')
    ndim = dispatch_property('ndim')
    dtype = dispatch_property('dtype')

    def __getitem__(self, key):
        return type(self)(self.arr[key], **self.attrs)

    def __setitem__(self, key, value):
        self.arr[key] = value