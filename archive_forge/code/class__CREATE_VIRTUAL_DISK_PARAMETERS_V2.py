import ctypes
from os_win.utils.winapi import wintypes
class _CREATE_VIRTUAL_DISK_PARAMETERS_V2(ctypes.Structure):
    _fields_ = [('UniqueId', wintypes.GUID), ('MaximumSize', wintypes.ULONGLONG), ('BlockSizeInBytes', wintypes.ULONG), ('SectorSizeInBytes', wintypes.ULONG), ('PhysicalSectorSizeInBytes', wintypes.ULONG), ('ParentPath', wintypes.LPCWSTR), ('SourcePath', wintypes.LPCWSTR), ('OpenFlags', wintypes.DWORD), ('ParentVirtualStorageType', VIRTUAL_STORAGE_TYPE), ('SourceVirtualStorageType', VIRTUAL_STORAGE_TYPE), ('ResiliencyGuid', wintypes.GUID)]