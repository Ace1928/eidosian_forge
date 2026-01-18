from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
class SYM_INFOW(Structure):
    _fields_ = [('SizeOfStruct', ULONG), ('TypeIndex', ULONG), ('Reserved', ULONG64 * 2), ('Index', ULONG), ('Size', ULONG), ('ModBase', ULONG64), ('Flags', ULONG), ('Value', ULONG64), ('Address', ULONG64), ('Register', ULONG), ('Scope', ULONG), ('Tag', ULONG), ('NameLen', ULONG), ('MaxNameLen', ULONG), ('Name', WCHAR * (MAX_SYM_NAME + 1))]