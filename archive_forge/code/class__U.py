import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class _U(Union):
    _anonymous_ = ('_buttons',)
    _fields_ = [('ulButtons', ULONG), ('_buttons', _Buttons)]