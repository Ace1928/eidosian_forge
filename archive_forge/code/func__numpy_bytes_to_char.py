from __future__ import annotations
from functools import partial
import numpy as np
from xarray.coding.variables import (
from xarray.core import indexing
from xarray.core.utils import module_available
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def _numpy_bytes_to_char(arr):
    """Like netCDF4.stringtochar, but faster and more flexible."""
    copy = None if HAS_NUMPY_2_0 else False
    arr = np.array(arr, copy=copy, order='C', dtype=np.bytes_)
    return arr.reshape(arr.shape + (1,)).view('S1')