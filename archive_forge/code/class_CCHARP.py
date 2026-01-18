import unittest
from ctypes.test import need_symbol
import test.support
class CCHARP(c_char_p):

    def from_param(cls, value):
        return value * 4
    from_param = classmethod(from_param)