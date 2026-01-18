import unittest
from ctypes import *
class cell(Structure):
    _fields_ = [('name', c_char_p), ('next', lpcell)]