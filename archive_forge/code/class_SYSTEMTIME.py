import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class SYSTEMTIME(Structure):
    _fields_ = [('wYear', WORD), ('wMonth', WORD), ('wDayOfWeek', WORD), ('wDay', WORD), ('wHour', WORD), ('wMinute', WORD), ('wSecond', WORD), ('wMilliseconds', WORD)]