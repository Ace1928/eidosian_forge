import unittest
from ctypes import *
from ctypes.test import need_symbol
def _check_retval_(value):
    return str(value.value)