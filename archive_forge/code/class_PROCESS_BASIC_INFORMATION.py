from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
class PROCESS_BASIC_INFORMATION(Structure):
    _fields_ = [('ExitStatus', SIZE_T), ('PebBaseAddress', PVOID), ('AffinityMask', KAFFINITY), ('BasePriority', SDWORD), ('UniqueProcessId', ULONG_PTR), ('InheritedFromUniqueProcessId', ULONG_PTR)]