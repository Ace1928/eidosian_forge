import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
class U1(_Union):
    _fields_ = [('s1', S1), ('ab', c_short)]