import os
import sys
from contextlib import contextmanager
class CursorInfo(ctypes.Structure):
    _fields_ = [('size', ctypes.c_int), ('visible', ctypes.c_byte)]