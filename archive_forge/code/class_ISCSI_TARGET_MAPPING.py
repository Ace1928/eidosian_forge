import ctypes
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
class ISCSI_TARGET_MAPPING(ctypes.Structure):
    _fields_ = [('InitiatorName', wintypes.WCHAR * w_const.MAX_ISCSI_HBANAME_LEN), ('TargetName', wintypes.WCHAR * (w_const.MAX_ISCSI_NAME_LEN + 1)), ('OSDeviceName', wintypes.WCHAR * w_const.MAX_PATH), ('SessionId', ISCSI_UNIQUE_SESSION_ID), ('OSBusNumber', wintypes.ULONG), ('OSTargetNumber', wintypes.ULONG), ('LUNCount', wintypes.ULONG), ('LUNList', PSCSI_LUN_LIST)]