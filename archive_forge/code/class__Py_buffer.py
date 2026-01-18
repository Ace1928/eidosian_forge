import os
import abc
import sys
from Cryptodome.Util.py3compat import byte_string
from Cryptodome.Util._file_system import pycryptodome_filename
class _Py_buffer(ctypes.Structure):
    _fields_ = [('buf', c_void_p), ('obj', ctypes.py_object), ('len', _c_ssize_t), ('itemsize', _c_ssize_t), ('readonly', ctypes.c_int), ('ndim', ctypes.c_int), ('format', ctypes.c_char_p), ('shape', _c_ssize_p), ('strides', _c_ssize_p), ('suboffsets', _c_ssize_p), ('internal', c_void_p)]
    if sys.version_info[0] == 2:
        _fields_.insert(-1, ('smalltable', _c_ssize_t * 2))