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
class TestRecursiveStructure(unittest.TestCase):

    def test_contains_itself(self):

        class Recursive(Structure):
            pass
        try:
            Recursive._fields_ = [('next', Recursive)]
        except AttributeError as details:
            self.assertIn('Structure or union cannot contain itself', str(details))
        else:
            self.fail('Structure or union cannot contain itself')

    def test_vice_versa(self):

        class First(Structure):
            pass

        class Second(Structure):
            pass
        First._fields_ = [('second', Second)]
        try:
            Second._fields_ = [('first', First)]
        except AttributeError as details:
            self.assertIn('_fields_ is final', str(details))
        else:
            self.fail('AttributeError not raised')