from winappdbg.win32.defines import *
def GetFileVersionInfoA(lptstrFilename):
    _GetFileVersionInfoA = windll.version.GetFileVersionInfoA
    _GetFileVersionInfoA.argtypes = [LPSTR, DWORD, DWORD, LPVOID]
    _GetFileVersionInfoA.restype = bool
    _GetFileVersionInfoA.errcheck = RaiseIfZero
    _GetFileVersionInfoSizeA = windll.version.GetFileVersionInfoSizeA
    _GetFileVersionInfoSizeA.argtypes = [LPSTR, LPVOID]
    _GetFileVersionInfoSizeA.restype = DWORD
    _GetFileVersionInfoSizeA.errcheck = RaiseIfZero
    dwLen = _GetFileVersionInfoSizeA(lptstrFilename, None)
    lpData = ctypes.create_string_buffer(dwLen)
    _GetFileVersionInfoA(lptstrFilename, 0, dwLen, byref(lpData))
    return lpData