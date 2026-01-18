from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def SHGetFolderPathW(nFolder, hToken=None, dwFlags=SHGFP_TYPE_CURRENT):
    _SHGetFolderPathW = windll.shell32.SHGetFolderPathW
    _SHGetFolderPathW.argtypes = [HWND, ctypes.c_int, HANDLE, DWORD, LPWSTR]
    _SHGetFolderPathW.restype = HRESULT
    _SHGetFolderPathW.errcheck = RaiseIfNotZero
    pszPath = ctypes.create_unicode_buffer(MAX_PATH + 1)
    _SHGetFolderPathW(None, nFolder, hToken, dwFlags, pszPath)
    return pszPath.value