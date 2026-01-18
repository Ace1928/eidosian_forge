import ctypes
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
class ISCSI_UNIQUE_SESSION_ID(ctypes.Structure):
    _fields_ = [('AdapterUnique', wintypes.ULONGLONG), ('AdapterSpecific', wintypes.ULONGLONG)]