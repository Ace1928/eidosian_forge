import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class _DUMMYSTRUCTNAME(Structure):
    _fields_ = [('dmOrientation', c_short), ('dmPaperSize', c_short), ('dmPaperLength', c_short), ('dmPaperWidth', c_short), ('dmScale', c_short), ('dmCopies', c_short), ('dmDefaultSource', c_short), ('dmPrintQuality', c_short)]