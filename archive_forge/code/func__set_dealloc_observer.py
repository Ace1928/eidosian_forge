import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def _set_dealloc_observer(objc_ptr):
    observer = send_message('DeallocationObserver', 'alloc')
    observer = send_message(observer, 'initWithObject:', objc_ptr, argtypes=_dealloc_argtype)
    objc.objc_setAssociatedObject(objc_ptr, observer, observer, OBJC_ASSOCIATION_RETAIN)
    send_message(observer, 'release')