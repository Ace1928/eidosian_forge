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
class U(Union):
    _fields_ = [('f1', c_uint8 * 16), ('f2', c_uint16 * 8), ('f3', c_uint32 * 4)]