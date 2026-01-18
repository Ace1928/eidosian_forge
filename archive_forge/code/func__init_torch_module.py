import ctypes
import sys
from .base import _LIB
from .base import c_array, c_str_array, c_handle_array, py_str, build_param_doc as _build_param_doc
from .base import mx_uint, mx_float, FunctionHandle
from .base import check_call
from .ndarray import NDArray, _new_empty_handle
def _init_torch_module():
    """List and add all the torch backed ndarray functions to current module."""
    plist = ctypes.POINTER(FunctionHandle)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListFunctions(ctypes.byref(size), ctypes.byref(plist)))
    module_obj = sys.modules[__name__]
    for i in range(size.value):
        hdl = FunctionHandle(plist[i])
        function = _make_torch_function(hdl)
        if function is not None:
            setattr(module_obj, function.__name__, function)