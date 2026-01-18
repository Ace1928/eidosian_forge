import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class EXCEPTION_RECORD64(Structure):
    _fields_ = [('ExceptionCode', DWORD), ('ExceptionFlags', DWORD), ('ExceptionRecord', DWORD64), ('ExceptionAddress', DWORD64), ('NumberParameters', DWORD), ('__unusedAlignment', DWORD), ('ExceptionInformation', DWORD64 * EXCEPTION_MAXIMUM_PARAMETERS)]