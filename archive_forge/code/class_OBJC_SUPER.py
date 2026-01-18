import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
class OBJC_SUPER(Structure):
    _fields_ = [('receiver', c_void_p), ('class', c_void_p)]