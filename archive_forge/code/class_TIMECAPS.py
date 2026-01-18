import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class TIMECAPS(Structure):
    _fields_ = (('wPeriodMin', UINT), ('wPeriodMax', UINT))