from __future__ import annotations
import json
import os
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray import coding, conventions
from xarray.backends.common import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import guess_chunkmanager
from xarray.namedarray.pycompat import integer_types
class ZarrArrayWrapper(BackendArray):
    __slots__ = ('dtype', 'shape', '_array')

    def __init__(self, zarr_array):
        self._array = zarr_array
        self.shape = self._array.shape
        if self._array.filters is not None and any([filt.codec_id == 'vlen-utf8' for filt in self._array.filters]):
            dtype = coding.strings.create_vlen_dtype(str)
        else:
            dtype = self._array.dtype
        self.dtype = dtype

    def get_array(self):
        return self._array

    def _oindex(self, key):
        return self._array.oindex[key]

    def _vindex(self, key):
        return self._array.vindex[key]

    def _getitem(self, key):
        return self._array[key]

    def __getitem__(self, key):
        array = self._array
        if isinstance(key, indexing.BasicIndexer):
            method = self._getitem
        elif isinstance(key, indexing.VectorizedIndexer):
            method = self._vindex
        elif isinstance(key, indexing.OuterIndexer):
            method = self._oindex
        return indexing.explicit_indexing_adapter(key, array.shape, indexing.IndexingSupport.VECTORIZED, method)