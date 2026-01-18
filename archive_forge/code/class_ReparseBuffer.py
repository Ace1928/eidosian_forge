import os
import pathlib
import platform
import stat
import sys
from logging import getLogger
from typing import Union
class ReparseBuffer(ctypes.Structure):
    _anonymous_ = ('u',)
    _fields_ = [('reparse_tag', ctypes.c_ulong), ('reparse_data_length', ctypes.c_ushort), ('reserved', ctypes.c_ushort), ('substitute_name_offset', ctypes.c_ushort), ('substitute_name_length', ctypes.c_ushort), ('print_name_offset', ctypes.c_ushort), ('print_name_length', ctypes.c_ushort), ('u', ReparseBufferField)]