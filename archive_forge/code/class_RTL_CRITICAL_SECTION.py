from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class RTL_CRITICAL_SECTION(Structure):
    _fields_ = [('DebugInfo', PVOID), ('LockCount', LONG), ('RecursionCount', LONG), ('OwningThread', PVOID), ('LockSemaphore', PVOID), ('SpinCount', ULONG)]