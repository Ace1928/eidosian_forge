import ctypes
from ...base import _LIB, check_call
from . import function
from .types import RETURN_SWITCH, TypeCode
def _return_object(x):
    handle = x.v_handle
    if not isinstance(handle, ObjectHandle):
        handle = ObjectHandle(handle)
    cls = function._CLASS_OBJECT
    obj = cls.__new__(cls)
    obj.handle = handle
    return obj