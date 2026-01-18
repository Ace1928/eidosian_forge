import ctypes
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
class PERSISTENT_ISCSI_LOGIN_INFO(ctypes.Structure):
    _fields_ = [('TargetName', wintypes.WCHAR * (w_const.MAX_ISCSI_NAME_LEN + 1)), ('IsInformationalSession', wintypes.BOOLEAN), ('InitiatorInstance', wintypes.WCHAR * w_const.MAX_ISCSI_HBANAME_LEN), ('InitiatorPortNumber', wintypes.ULONG), ('TargetPortal', ISCSI_TARGET_PORTAL), ('SecurityFlags', ISCSI_SECURITY_FLAGS), ('Mappings', PISCSI_TARGET_MAPPING), ('LoginOptions', ISCSI_LOGIN_OPTIONS)]