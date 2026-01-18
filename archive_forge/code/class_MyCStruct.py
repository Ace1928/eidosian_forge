import unittest
from ctypes import *
class MyCStruct(Structure):
    _fields_ = (('field', c_int),)