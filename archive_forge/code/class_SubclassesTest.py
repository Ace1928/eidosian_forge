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
class SubclassesTest(unittest.TestCase):

    def test_subclass(self):

        class X(Structure):
            _fields_ = [('a', c_int)]

        class Y(X):
            _fields_ = [('b', c_int)]

        class Z(X):
            pass
        self.assertEqual(sizeof(X), sizeof(c_int))
        self.assertEqual(sizeof(Y), sizeof(c_int) * 2)
        self.assertEqual(sizeof(Z), sizeof(c_int))
        self.assertEqual(X._fields_, [('a', c_int)])
        self.assertEqual(Y._fields_, [('b', c_int)])
        self.assertEqual(Z._fields_, [('a', c_int)])

    def test_subclass_delayed(self):

        class X(Structure):
            pass
        self.assertEqual(sizeof(X), 0)
        X._fields_ = [('a', c_int)]

        class Y(X):
            pass
        self.assertEqual(sizeof(Y), sizeof(X))
        Y._fields_ = [('b', c_int)]

        class Z(X):
            pass
        self.assertEqual(sizeof(X), sizeof(c_int))
        self.assertEqual(sizeof(Y), sizeof(c_int) * 2)
        self.assertEqual(sizeof(Z), sizeof(c_int))
        self.assertEqual(X._fields_, [('a', c_int)])
        self.assertEqual(Y._fields_, [('b', c_int)])
        self.assertEqual(Z._fields_, [('a', c_int)])