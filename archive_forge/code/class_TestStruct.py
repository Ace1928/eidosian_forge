import unittest
import ctypes
from ctypes.test import need_symbol
import _ctypes_test
class TestStruct(ctypes.Structure):
    _fields_ = [('unicode', ctypes.c_wchar_p)]