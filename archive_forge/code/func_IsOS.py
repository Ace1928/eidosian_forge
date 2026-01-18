from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def IsOS(dwOS):
    try:
        _IsOS = windll.shlwapi.IsOS
        _IsOS.argtypes = [DWORD]
        _IsOS.restype = bool
    except AttributeError:
        _GetProcAddress = windll.kernel32.GetProcAddress
        _GetProcAddress.argtypes = [HINSTANCE, DWORD]
        _GetProcAddress.restype = LPVOID
        _IsOS = windll.kernel32.GetProcAddress(windll.shlwapi._handle, 437)
        _IsOS = WINFUNCTYPE(bool, DWORD)(_IsOS)
    return _IsOS(dwOS)