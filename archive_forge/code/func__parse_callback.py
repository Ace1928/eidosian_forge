from . import _ccallback_c
import ctypes
@classmethod
def _parse_callback(cls, obj, user_data=None, signature=None):
    _import_cffi()
    if isinstance(obj, LowLevelCallable):
        func = tuple.__getitem__(obj, 0)
    elif isinstance(obj, PyCFuncPtr):
        func, signature = _get_ctypes_func(obj, signature)
    elif isinstance(obj, CData):
        func, signature = _get_cffi_func(obj, signature)
    elif _ccallback_c.check_capsule(obj):
        func = obj
    else:
        raise ValueError('Given input is not a callable or a low-level callable (pycapsule/ctypes/cffi)')
    if isinstance(user_data, ctypes.c_void_p):
        context = _get_ctypes_data(user_data)
    elif isinstance(user_data, CData):
        context = _get_cffi_data(user_data)
    elif user_data is None:
        context = 0
    elif _ccallback_c.check_capsule(user_data):
        context = user_data
    else:
        raise ValueError('Given user data is not a valid low-level void* pointer (pycapsule/ctypes/cffi)')
    return _ccallback_c.get_raw_capsule(func, signature, context)