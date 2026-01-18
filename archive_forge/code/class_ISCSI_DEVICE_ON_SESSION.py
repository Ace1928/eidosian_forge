import ctypes
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
class ISCSI_DEVICE_ON_SESSION(ctypes.Structure):
    _fields_ = [('InitiatorName', wintypes.WCHAR * w_const.MAX_ISCSI_HBANAME_LEN), ('TargetName', wintypes.WCHAR * (w_const.MAX_ISCSI_NAME_LEN + 1)), ('ScsiAddress', SCSI_ADDRESS), ('DeviceInterfaceType', wintypes.GUID), ('DeviceInterfaceName', wintypes.WCHAR * w_const.MAX_PATH), ('LegacyName', wintypes.WCHAR * w_const.MAX_PATH), ('StorageDeviceNumber', STORAGE_DEVICE_NUMBER), ('DeviceInstance', wintypes.ULONG)]