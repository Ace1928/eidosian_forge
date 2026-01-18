import ctypes
import warnings
import operator
from array import array as native_array
import numpy as np
from ..base import NotSupportedForSparseNDArray
from ..base import _LIB, numeric_types
from ..base import c_array_buf, mx_real_t, integer_types
from ..base import NDArrayHandle, check_call
from ..context import Context, current_context
from . import _internal
from . import op
from ._internal import _set_ndarray_class
from .ndarray import NDArray, _storage_type, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ROW_SPARSE, _STORAGE_TYPE_CSR, _int64_enabled
from .ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray import zeros as _zeros_ndarray
from .ndarray import array as _array
from .ndarray import _ufunc_helper
def _aux_data(self, i):
    """ Get a deep copy NDArray of the i-th aux data array associated with the
        BaseSparseNDArray.

        This function blocks. Do not use it in performance critical code.
        """
    self.wait_to_read()
    hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayGetAuxNDArray(self.handle, i, ctypes.byref(hdl)))
    return NDArray(hdl)