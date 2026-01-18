import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
class AsParamWrapperTestCase(BasicWrapTestCase):
    wrap = AsParamWrapper