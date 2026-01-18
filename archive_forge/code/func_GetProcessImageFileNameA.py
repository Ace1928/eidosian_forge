from winappdbg.win32.defines import *
def GetProcessImageFileNameA(hProcess):
    _GetProcessImageFileNameA = windll.psapi.GetProcessImageFileNameA
    _GetProcessImageFileNameA.argtypes = [HANDLE, LPSTR, DWORD]
    _GetProcessImageFileNameA.restype = DWORD
    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_string_buffer('', nSize)
        nCopied = _GetProcessImageFileNameA(hProcess, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < nSize - 1:
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value