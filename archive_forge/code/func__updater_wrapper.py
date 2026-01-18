import pickle
import ctypes
import os
from ..ndarray import NDArray
from ..ndarray import _ndarray_cls
from ..base import _LIB, c_str
from ..base import check_call, mx_uint, py_str
from ..base import NDArrayHandle, KVStoreHandle
from .. import optimizer as opt
from .base import _ctype_key_value, _ctype_dict, KVStoreBase
def _updater_wrapper(updater):
    """A wrapper for the user-defined handle."""

    def updater_handle(key, lhs_handle, rhs_handle, _):
        """ ctypes function """
        lhs = _ndarray_cls(NDArrayHandle(lhs_handle))
        rhs = _ndarray_cls(NDArrayHandle(rhs_handle))
        updater(key, lhs, rhs)
    return updater_handle