from winappdbg.win32.defines import *
from winappdbg.win32.advapi32 import *
def WTSFreeMemory(pMemory):
    _WTSFreeMemory = windll.wtsapi32.WTSFreeMemory
    _WTSFreeMemory.argtypes = [PVOID]
    _WTSFreeMemory.restype = None
    _WTSFreeMemory(pMemory)