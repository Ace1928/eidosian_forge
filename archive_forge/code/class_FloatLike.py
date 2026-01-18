from ctypes import *
import unittest
import struct
class FloatLike:

    def __float__(self):
        return 2.0