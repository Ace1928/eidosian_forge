import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class RAWINPUT(Structure):
    _fields_ = [('header', RAWINPUTHEADER), ('data', _RAWINPUTDEVICEUNION)]