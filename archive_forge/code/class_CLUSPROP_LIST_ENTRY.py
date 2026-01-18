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
class CLUSPROP_LIST_ENTRY(ctypes.Structure):
    _fields_ = [('name', self._get_clusprop_value_struct(val_type=ctypes.c_wchar * name_len)), ('value', self._get_clusprop_value_struct(val_type=ctypes.c_ubyte * val_sz)), ('_endmark', wintypes.DWORD)]