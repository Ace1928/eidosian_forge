import ctypes
from os_win.utils.winapi import wintypes
class _VHD_INFO_PHYSICAL_DISK(ctypes.Structure):
    _fields_ = [('LogicalSectorSize', wintypes.ULONG), ('PhysicalSectorSize', wintypes.ULONG), ('IsRemote', wintypes.BOOL)]