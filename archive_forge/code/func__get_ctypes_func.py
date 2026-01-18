from . import _ccallback_c
import ctypes
def _get_ctypes_func(func, signature=None):
    func_ptr = ctypes.cast(func, ctypes.c_void_p).value
    if signature is None:
        signature = _typename_from_ctypes(func.restype) + ' ('
        for j, arg in enumerate(func.argtypes):
            if j == 0:
                signature += _typename_from_ctypes(arg)
            else:
                signature += ', ' + _typename_from_ctypes(arg)
        signature += ')'
    return (func_ptr, signature)