import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def QueryDosDeviceW(lpDeviceName):
    _QueryDosDeviceW = windll.kernel32.QueryDosDeviceW
    _QueryDosDeviceW.argtypes = [LPWSTR, LPWSTR, DWORD]
    _QueryDosDeviceW.restype = DWORD
    _QueryDosDeviceW.errcheck = RaiseIfZero
    if not lpDeviceName:
        lpDeviceName = None
    ucchMax = 4096
    lpTargetPath = ctypes.create_unicode_buffer(u'', ucchMax)
    _QueryDosDeviceW(lpDeviceName, lpTargetPath, ucchMax)
    return lpTargetPath.value