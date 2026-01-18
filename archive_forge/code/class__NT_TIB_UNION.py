from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class _NT_TIB_UNION(Union):
    _fields_ = [('FiberData', PVOID), ('Version', ULONG)]