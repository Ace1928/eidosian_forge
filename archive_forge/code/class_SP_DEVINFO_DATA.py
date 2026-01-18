from __future__ import absolute_import
import re
import ctypes
from ctypes.wintypes import BOOL
from ctypes.wintypes import HWND
from ctypes.wintypes import DWORD
from ctypes.wintypes import WORD
from ctypes.wintypes import LONG
from ctypes.wintypes import ULONG
from ctypes.wintypes import HKEY
from ctypes.wintypes import BYTE
import serial
from serial.win32 import ULONG_PTR
from serial.tools import list_ports_common
class SP_DEVINFO_DATA(ctypes.Structure):
    _fields_ = [('cbSize', DWORD), ('ClassGuid', GUID), ('DevInst', DWORD), ('Reserved', ULONG_PTR)]

    def __str__(self):
        return 'ClassGuid:{} DevInst:{}'.format(self.ClassGuid, self.DevInst)