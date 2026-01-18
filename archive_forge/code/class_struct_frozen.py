import _imp
import importlib.util
import unittest
import sys
from ctypes import *
from test.support import import_helper
import _ctypes_test
class struct_frozen(Structure):
    _fields_ = [('name', c_char_p), ('code', POINTER(c_ubyte)), ('size', c_int), ('is_package', c_int), ('get_code', POINTER(c_ubyte))]