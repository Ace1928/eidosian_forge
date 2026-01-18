from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
class Little(LittleEndianStructure):
    _fields_ = [('a', c_uint32, 24), ('b', c_uint32, 4), ('c', c_uint32, 4)]