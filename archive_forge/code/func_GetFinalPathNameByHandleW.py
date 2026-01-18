import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetFinalPathNameByHandleW(hFile, dwFlags=FILE_NAME_NORMALIZED | VOLUME_NAME_DOS):
    _GetFinalPathNameByHandleW = windll.kernel32.GetFinalPathNameByHandleW
    _GetFinalPathNameByHandleW.argtypes = [HANDLE, LPWSTR, DWORD, DWORD]
    _GetFinalPathNameByHandleW.restype = DWORD
    cchFilePath = _GetFinalPathNameByHandleW(hFile, None, 0, dwFlags)
    if cchFilePath == 0:
        raise ctypes.WinError()
    lpszFilePath = ctypes.create_unicode_buffer(u'', cchFilePath + 1)
    nCopied = _GetFinalPathNameByHandleW(hFile, lpszFilePath, cchFilePath, dwFlags)
    if nCopied <= 0 or nCopied > cchFilePath:
        raise ctypes.WinError()
    return lpszFilePath.value