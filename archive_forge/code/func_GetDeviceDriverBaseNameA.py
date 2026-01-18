from winappdbg.win32.defines import *
def GetDeviceDriverBaseNameA(ImageBase):
    _GetDeviceDriverBaseNameA = windll.psapi.GetDeviceDriverBaseNameA
    _GetDeviceDriverBaseNameA.argtypes = [LPVOID, LPSTR, DWORD]
    _GetDeviceDriverBaseNameA.restype = DWORD
    nSize = MAX_PATH
    while 1:
        lpBaseName = ctypes.create_string_buffer('', nSize)
        nCopied = _GetDeviceDriverBaseNameA(ImageBase, lpBaseName, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < nSize - 1:
            break
        nSize = nSize + MAX_PATH
    return lpBaseName.value