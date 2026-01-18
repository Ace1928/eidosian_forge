import socket
import os
import sys
class in6_addr(ctypes.Structure):
    _fields_ = [('Byte', ctypes.c_ubyte * 16)]