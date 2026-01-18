from array import array as native_array
import ctypes
import warnings
import numpy as _np
from ..autograd import is_recording
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _GRAD_REQ_MAP
from ..ndarray import indexing_key_expand_implicit_axes, get_indexing_dispatch_code,\
from ..ndarray._internal import _set_np_ndarray_class
from . import _op as _mx_np_op
from ..base import check_call, _LIB, NDArrayHandle, c_array
from ..base import mx_real_t, c_array_buf, mx_uint, numeric_types, integer_types
from ..context import Context
from ..util import set_module, wrap_np_unary_func, wrap_np_binary_func
from ..context import current_context
from ..ndarray import numpy as _mx_nd_np
from ..ndarray.numpy import _internal as _npi
from ..ndarray.ndarray import _storage_type, from_numpy
from .utils import _get_np_op
from .fallback import *  # pylint: disable=wildcard-import,unused-wildcard-import
from . import fallback
def _get_np_empty_tuple_indexing(self, key):
    new_shape = []
    num_none = 0
    for i, idx in enumerate(key):
        if idx is None:
            new_shape.append(1)
            num_none += 1
        elif idx == ():
            new_shape.append(0)
        elif idx == slice(None, None, None):
            new_shape.append(self.shape[i - num_none])
    return empty(new_shape, dtype=self.dtype)