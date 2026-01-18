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
def _as_mx_np_array(object, ctx=None):
    """Convert object to mxnet.numpy.ndarray."""
    if isinstance(object, _np.ndarray):
        if not object.flags['C_CONTIGUOUS']:
            object = _np.ascontiguousarray(object, dtype=object.dtype)
        ret = from_numpy(object, array_cls=ndarray)
        return ret if ctx is None else ret.as_in_ctx(ctx=ctx)
    elif isinstance(object, (integer_types, numeric_types)):
        return object
    elif isinstance(object, (list, tuple)):
        tmp = [_as_mx_np_array(arr) for arr in object]
        return object.__class__(tmp)
    elif isinstance(object, (_np.bool_, _np.bool)):
        return array(object, dtype=_np.bool_, ctx=ctx)
    else:
        raise TypeError('Does not support converting {} to mx.np.ndarray.'.format(str(type(object))))