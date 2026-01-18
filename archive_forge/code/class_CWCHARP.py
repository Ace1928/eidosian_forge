import unittest
from ctypes.test import need_symbol
import test.support
class CWCHARP(c_wchar_p):

    def from_param(cls, value):
        return value * 3
    from_param = classmethod(from_param)