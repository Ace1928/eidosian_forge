import ctypes
from numbers import Number, Integral
from ...base import get_last_ffi_error, _LIB
from ..base import c_str
from .types import MXNetValue, TypeCode
from .types import RETURN_SWITCH
from .object import ObjectBase
from ..node_generic import convert_to_node
from ..._ctypes.ndarray import NDArrayBase
class FunctionBase(object):
    """Function base."""
    __slots__ = ['handle', 'is_global']

    def __init__(self, handle, is_global):
        """Initialize the function with handle

        Parameters
        ----------
        handle : FunctionHandle
            the handle to the underlying function.

        is_global : bool
            Whether this is a global function in python
        """
        self.handle = handle
        self.is_global = is_global

    def __del__(self):
        if not self.is_global and _LIB is not None:
            if _LIB.MXNetFuncFree(self.handle) != 0:
                raise get_last_ffi_error()

    def __call__(self, *args):
        """Call the function with positional arguments

        args : list
           The positional arguments to the function call.
        """
        temp_args = []
        values, tcodes, num_args = _make_mxnet_args(args, temp_args)
        ret_val = MXNetValue()
        ret_tcode = ctypes.c_int()
        if _LIB.MXNetFuncCall(self.handle, values, tcodes, ctypes.c_int(num_args), ctypes.byref(ret_val), ctypes.byref(ret_tcode)) != 0:
            raise get_last_ffi_error()
        _ = temp_args
        _ = args
        return RETURN_SWITCH[ret_tcode.value](ret_val)