import platform
from platform import architecture as _architecture
import struct
import sys
import unittest
from ctypes.test import need_symbol
from ctypes import (CDLL, Array, Structure, Union, POINTER, sizeof, byref, alignment,
from ctypes.util import find_library
from struct import calcsize
import _ctypes_test
from collections import namedtuple
from test import support
class PointerMemberTestCase(unittest.TestCase):

    def test(self):

        class S(Structure):
            _fields_ = [('array', POINTER(c_int))]
        s = S()
        s.array = (c_int * 3)(1, 2, 3)
        items = [s.array[i] for i in range(3)]
        self.assertEqual(items, [1, 2, 3])
        s.array[0] = 42
        items = [s.array[i] for i in range(3)]
        self.assertEqual(items, [42, 2, 3])
        s.array[0] = 1
        items = [s.array[i] for i in range(3)]
        self.assertEqual(items, [1, 2, 3])

    def test_none_to_pointer_fields(self):

        class S(Structure):
            _fields_ = [('x', c_int), ('p', POINTER(c_int))]
        s = S()
        s.x = 12345678
        s.p = None
        self.assertEqual(s.x, 12345678)