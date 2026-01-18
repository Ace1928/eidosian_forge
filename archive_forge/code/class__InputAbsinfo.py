import libevdev
import os
import ctypes
import errno
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_uint
from ctypes import c_void_p
from ctypes import c_long
from ctypes import c_int32
from ctypes import c_uint16
class _InputAbsinfo(ctypes.Structure):
    _fields_ = [('value', c_int32), ('minimum', c_int32), ('maximum', c_int32), ('fuzz', c_int32), ('flat', c_int32), ('resolution', c_int32)]