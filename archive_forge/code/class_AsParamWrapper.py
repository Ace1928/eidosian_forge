import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
class AsParamWrapper:

    def __init__(self, param):
        self._as_parameter_ = param