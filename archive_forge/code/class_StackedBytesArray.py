from __future__ import annotations
from functools import partial
import numpy as np
from xarray.coding.variables import (
from xarray.core import indexing
from xarray.core.utils import module_available
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
class StackedBytesArray(indexing.ExplicitlyIndexedNDArrayMixin):
    """Wrapper around array-like objects to create a new indexable object where
    values, when accessed, are automatically stacked along the last dimension.

    >>> indexer = indexing.BasicIndexer((slice(None),))
    >>> StackedBytesArray(np.array(["a", "b", "c"], dtype="S1"))[indexer]
    array(b'abc', dtype='|S3')
    """

    def __init__(self, array):
        """
        Parameters
        ----------
        array : array-like
            Original array of values to wrap.
        """
        if array.dtype != 'S1':
            raise ValueError("can only use StackedBytesArray if argument has dtype='S1'")
        self.array = indexing.as_indexable(array)

    @property
    def dtype(self):
        return np.dtype('S' + str(self.array.shape[-1]))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape[:-1]

    def __repr__(self):
        return f'{type(self).__name__}({self.array!r})'

    def _vindex_get(self, key):
        return _numpy_char_to_bytes(self.array.vindex[key])

    def _oindex_get(self, key):
        return _numpy_char_to_bytes(self.array.oindex[key])

    def __getitem__(self, key):
        key = type(key)(indexing.expanded_indexer(key.tuple, self.array.ndim))
        if key.tuple[-1] != slice(None):
            raise IndexError('too many indices')
        return _numpy_char_to_bytes(self.array[key])