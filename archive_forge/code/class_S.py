from ctypes import *
import unittest, sys
from test import support
import _ctypes_test
class S(Structure):
    _fields_ = fields