import unittest
from ctypes.test import need_symbol
import test.support
class CVOIDP(c_void_p):

    def from_param(cls, value):
        return value * 2
    from_param = classmethod(from_param)