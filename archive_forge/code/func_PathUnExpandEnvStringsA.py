from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathUnExpandEnvStringsA(pszPath):
    _PathUnExpandEnvStringsA = windll.shlwapi.PathUnExpandEnvStringsA
    _PathUnExpandEnvStringsA.argtypes = [LPSTR, LPSTR]
    _PathUnExpandEnvStringsA.restype = bool
    _PathUnExpandEnvStringsA.errcheck = RaiseIfZero
    cchBuf = MAX_PATH
    pszBuf = ctypes.create_string_buffer('', cchBuf)
    _PathUnExpandEnvStringsA(pszPath, pszBuf, cchBuf)
    return pszBuf.value