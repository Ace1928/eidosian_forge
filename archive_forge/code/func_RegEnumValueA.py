from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegEnumValueA(hKey, dwIndex, bGetData=True):
    return _internal_RegEnumValue(True, hKey, dwIndex, bGetData)