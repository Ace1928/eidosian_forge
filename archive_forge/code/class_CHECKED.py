import unittest
from ctypes import *
from ctypes.test import need_symbol
class CHECKED(c_int):

    def _check_retval_(value):
        return str(value.value)
    _check_retval_ = staticmethod(_check_retval_)