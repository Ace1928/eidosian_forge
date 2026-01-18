import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
class Nasty:

    def __del__(self):
        gc.collect()