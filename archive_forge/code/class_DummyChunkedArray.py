from __future__ import annotations
from importlib.metadata import EntryPoint
from typing import Any
import numpy as np
import pytest
from xarray.core.types import T_Chunks, T_DuckArray, T_NormalizedChunks
from xarray.namedarray._typing import _Chunks
from xarray.namedarray.daskmanager import DaskManager
from xarray.namedarray.parallelcompat import (
from xarray.tests import has_dask, requires_dask
class DummyChunkedArray(np.ndarray):
    """
    Mock-up of a chunked array class.

    Adds a (non-functional) .chunks attribute by following this example in the numpy docs
    https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    """
    chunks: T_NormalizedChunks

    def __new__(cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, chunks=None):
        obj = super().__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.chunks = chunks
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.chunks = getattr(obj, 'chunks', None)

    def rechunk(self, chunks, **kwargs):
        copied = self.copy()
        copied.chunks = chunks
        return copied