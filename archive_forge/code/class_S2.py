import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
class S2(_Structure):
    _fields_ = [('u1', U1), ('c', c_byte)]