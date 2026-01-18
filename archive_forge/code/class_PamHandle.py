import os
import six
import sys
from ctypes import cdll
from ctypes import CFUNCTYPE
from ctypes import CDLL
from ctypes import POINTER
from ctypes import Structure
from ctypes import byref
from ctypes import cast
from ctypes import sizeof
from ctypes import py_object
from ctypes import c_char
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_size_t
from ctypes import c_void_p
from ctypes import memmove
from ctypes.util import find_library
from typing import Union
class PamHandle(Structure):
    """wrapper class for pam_handle_t pointer"""
    _fields_ = [('handle', c_void_p)]

    def __init__(self):
        super().__init__()
        self.handle = 0

    def __repr__(self):
        return f'<PamHandle {self.handle}>'