import ctypes
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
class SCSI_ADDRESS(ctypes.Structure):
    _fields_ = [('Length', wintypes.ULONG), ('PortNumber', ctypes.c_ubyte), ('PathId', ctypes.c_ubyte), ('TargetId', ctypes.c_ubyte), ('Lun', ctypes.c_ubyte)]