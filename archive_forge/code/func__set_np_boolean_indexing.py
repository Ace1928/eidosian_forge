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
def _set_np_boolean_indexing(self, key, value):
    """
        There are two types of boolean indices (which are equivalent,
        for the most part though). This function will handle single boolean assign for higher speed.
        If this is not the case, it is instead expanded into (multiple)
        integer array indices and will be handled by advanced assign.
        """
    if isinstance(value, numeric_types):
        _npi.boolean_mask_assign_scalar(data=self, mask=key, value=int(value) if isinstance(value, bool) else value, start_axis=0, out=self)
    elif isinstance(value, ndarray):
        _npi.boolean_mask_assign_tensor(data=self, mask=key, value=value, start_axis=0, out=self)
    else:
        raise NotImplementedError('type %s is not supported.' % type(value))