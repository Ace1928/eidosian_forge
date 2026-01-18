import ctypes
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
class ISCSI_TARGET_PORTAL(ctypes.Structure):
    _fields_ = [('SymbolicName', wintypes.WCHAR * w_const.MAX_ISCSI_PORTAL_NAME_LEN), ('Address', wintypes.WCHAR * w_const.MAX_ISCSI_PORTAL_ADDRESS_LEN), ('Socket', wintypes.USHORT)]