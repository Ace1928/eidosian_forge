from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
class S8I(Structure):
    _fields_ = [('a', c_int), ('b', c_int), ('c', c_int), ('d', c_int), ('e', c_int), ('f', c_int), ('g', c_int), ('h', c_int)]