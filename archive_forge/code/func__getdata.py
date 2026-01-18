from collections import namedtuple
import sys
import ctypes
import logging
import threading
import numpy as np
from ..base import _LIB
from ..base import c_str_array, mx_uint, py_str
from ..base import DataIterHandle, NDArrayHandle
from ..base import mx_real_t
from ..base import check_call, build_param_doc as _build_param_doc
from ..ndarray import NDArray
from ..ndarray.sparse import CSRNDArray
from ..ndarray import _ndarray_cls
from ..ndarray import array
from ..ndarray import concat, tile
from .utils import _init_data, _has_instance, _getdata_by_idx, _slice_along_batch_axis
def _getdata(self, data_source, start=None, end=None):
    """Load data from underlying arrays."""
    assert start is not None or end is not None, 'should at least specify start or end'
    start = start if start is not None else 0
    if end is None:
        end = data_source[0][1].shape[0] if data_source else 0
    s = slice(start, end)
    return [_slice_along_batch_axis(x[1], s, self.layout.find('N')) if isinstance(x[1], (np.ndarray, NDArray)) else array(x[1][sorted(self.idx[s])][[list(self.idx[s]).index(i) for i in sorted(self.idx[s])]]) for x in data_source]