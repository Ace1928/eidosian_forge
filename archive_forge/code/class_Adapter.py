import unittest
from ctypes.test import need_symbol
import test.support
class Adapter:

    def from_param(cls, obj):
        raise ValueError(obj)