from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class RTL_CRITICAL_SECTION_DEBUG(Structure):
    _fields_ = [('Type', WORD), ('CreatorBackTraceIndex', WORD), ('CriticalSection', PVOID), ('ProcessLocksList', LIST_ENTRY), ('EntryCount', ULONG), ('ContentionCount', ULONG), ('Flags', ULONG), ('CreatorBackTraceIndexHigh', WORD), ('SpareUSHORT', WORD)]