import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
class S1(_Structure):
    _fields_ = [('a', c_byte), ('b', c_byte)]