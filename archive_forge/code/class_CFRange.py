from ctypes import *
import sys, platform, struct
class CFRange(Structure):
    _fields_ = [('location', CFIndex), ('length', CFIndex)]