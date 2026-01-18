import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def S(self):
    return c_longlong.in_dll(self._dll, 'last_tf_arg_s').value