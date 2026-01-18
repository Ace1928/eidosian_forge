import ctypes
import os_win.conf
from os_win.utils.winapi import wintypes
class HBA_WWN(ctypes.Structure):
    _fields_ = [('wwn', ctypes.c_ubyte * 8)]