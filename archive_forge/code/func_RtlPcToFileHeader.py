import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def RtlPcToFileHeader(PcValue):
    _RtlPcToFileHeader = windll.kernel32.RtlPcToFileHeader
    _RtlPcToFileHeader.argtypes = [PVOID, POINTER(PVOID)]
    _RtlPcToFileHeader.restype = PRUNTIME_FUNCTION
    BaseOfImage = PVOID(0)
    _RtlPcToFileHeader(PcValue, byref(BaseOfImage))
    return BaseOfImage.value