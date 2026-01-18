from winappdbg.win32.defines import *
def GetFileVersionInfoW(lptstrFilename):
    _GetFileVersionInfoW = windll.version.GetFileVersionInfoW
    _GetFileVersionInfoW.argtypes = [LPWSTR, DWORD, DWORD, LPVOID]
    _GetFileVersionInfoW.restype = bool
    _GetFileVersionInfoW.errcheck = RaiseIfZero
    _GetFileVersionInfoSizeW = windll.version.GetFileVersionInfoSizeW
    _GetFileVersionInfoSizeW.argtypes = [LPWSTR, LPVOID]
    _GetFileVersionInfoSizeW.restype = DWORD
    _GetFileVersionInfoSizeW.errcheck = RaiseIfZero
    dwLen = _GetFileVersionInfoSizeW(lptstrFilename, None)
    lpData = ctypes.create_string_buffer(dwLen)
    _GetFileVersionInfoW(lptstrFilename, 0, dwLen, byref(lpData))
    return lpData