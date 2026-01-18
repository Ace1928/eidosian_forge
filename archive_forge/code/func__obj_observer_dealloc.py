import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def _obj_observer_dealloc(objc_obs, selector_name):
    """Removes any cached ObjCInstances in Python to prevent memory leaks.
    Manually break association as it's not implicitly mentioned that dealloc would break an association,
    although we do not use the object after.
    """
    objc_ptr = get_instance_variable(objc_obs, 'observed_object', c_void_p)
    if objc_ptr:
        objc.objc_setAssociatedObject(objc_ptr, objc_obs, None, OBJC_ASSOCIATION_ASSIGN)
        ObjCInstance._cached_objects.pop(objc_ptr, None)
    send_super(objc_obs, selector_name)