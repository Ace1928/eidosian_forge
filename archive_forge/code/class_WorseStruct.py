import unittest
from ctypes.test import need_symbol
import test.support
class WorseStruct(Structure):

    @property
    def __dict__(self):
        1 / 0