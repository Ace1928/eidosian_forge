import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
class OBJC_METHOD_DESCRIPTION(Structure):
    _fields_ = [('name', c_void_p), ('types', c_char_p)]