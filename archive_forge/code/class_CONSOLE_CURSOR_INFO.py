import ctypes
import sys
from typing import Any
import time
from ctypes import Structure, byref, wintypes
from typing import IO, NamedTuple, Type, cast
from pip._vendor.rich.color import ColorSystem
from pip._vendor.rich.style import Style
class CONSOLE_CURSOR_INFO(ctypes.Structure):
    _fields_ = [('dwSize', wintypes.DWORD), ('bVisible', wintypes.BOOL)]