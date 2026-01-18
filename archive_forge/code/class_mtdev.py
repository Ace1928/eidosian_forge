import os
import time
from ctypes import cdll, Structure, c_ulong, c_int, c_ushort, \
class mtdev(Structure):
    _fields_ = [('caps', mtdev_caps), ('state', c_void_p)]