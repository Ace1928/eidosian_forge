import ctypes
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
class ISCSI_SESSION_INFO(ctypes.Structure):
    _fields_ = [('SessionId', ISCSI_UNIQUE_SESSION_ID), ('InitiatorName', wintypes.PWSTR), ('TargetName', wintypes.PWSTR), ('TargetNodeName', wintypes.PWSTR), ('ISID', ctypes.c_ubyte * 6), ('TSID', ctypes.c_ubyte * 2), ('ConnectionCount', wintypes.ULONG), ('Connections', PISCSI_CONNECTION_INFO)]