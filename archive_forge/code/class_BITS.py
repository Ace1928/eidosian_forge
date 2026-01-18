from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
class BITS(Structure):
    _fields_ = [('A', c_int, 1), ('B', c_int, 2), ('C', c_int, 3), ('D', c_int, 4), ('E', c_int, 5), ('F', c_int, 6), ('G', c_int, 7), ('H', c_int, 8), ('I', c_int, 9), ('M', c_short, 1), ('N', c_short, 2), ('O', c_short, 3), ('P', c_short, 4), ('Q', c_short, 5), ('R', c_short, 6), ('S', c_short, 7)]