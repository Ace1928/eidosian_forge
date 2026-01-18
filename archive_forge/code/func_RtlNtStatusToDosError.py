from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
def RtlNtStatusToDosError(Status):
    _RtlNtStatusToDosError = windll.ntdll.RtlNtStatusToDosError
    _RtlNtStatusToDosError.argtypes = [NTSTATUS]
    _RtlNtStatusToDosError.restype = ULONG
    return _RtlNtStatusToDosError(Status)