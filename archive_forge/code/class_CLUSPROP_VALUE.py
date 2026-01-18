import contextlib
import ctypes
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
class CLUSPROP_VALUE(ctypes.Structure):
    _fields_ = [('syntax', wintypes.DWORD), ('length', wintypes.DWORD), ('value', val_type), ('_padding', ctypes.c_ubyte * _get_padding())]