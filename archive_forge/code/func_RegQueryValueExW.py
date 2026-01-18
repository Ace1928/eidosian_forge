from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegQueryValueExW(hKey, lpValueName=None, bGetData=True):
    return _internal_RegQueryValueEx(False, hKey, lpValueName, bGetData)