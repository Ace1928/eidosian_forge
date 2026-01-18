import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetLogicalDriveStringsW():
    _GetLogicalDriveStringsW = ctypes.windll.kernel32.GetLogicalDriveStringsW
    _GetLogicalDriveStringsW.argtypes = [DWORD, LPWSTR]
    _GetLogicalDriveStringsW.restype = DWORD
    _GetLogicalDriveStringsW.errcheck = RaiseIfZero
    nBufferLength = 4 * 26 + 1
    lpBuffer = ctypes.create_unicode_buffer(u'', nBufferLength)
    _GetLogicalDriveStringsW(nBufferLength, lpBuffer)
    drive_strings = list()
    string_p = addressof(lpBuffer)
    sizeof_wchar = sizeof(ctypes.c_wchar)
    while True:
        string_v = ctypes.wstring_at(string_p)
        if string_v == u'':
            break
        drive_strings.append(string_v)
        string_p += len(string_v) * sizeof_wchar + sizeof_wchar
    return drive_strings