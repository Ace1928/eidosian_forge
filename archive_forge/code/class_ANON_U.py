import unittest
import test.support
from ctypes import *
class ANON_U(Union):
    _fields_ = [('_', ANON_S), ('b', c_int)]
    _anonymous_ = ['_']