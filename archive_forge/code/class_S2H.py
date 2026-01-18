from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
class S2H(Structure):
    _fields_ = [('x', c_short), ('y', c_short)]