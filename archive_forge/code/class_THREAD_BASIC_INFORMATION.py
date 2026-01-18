from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
class THREAD_BASIC_INFORMATION(Structure):
    _fields_ = [('ExitStatus', NTSTATUS), ('TebBaseAddress', PVOID), ('ClientId', CLIENT_ID), ('AffinityMask', KAFFINITY), ('Priority', SDWORD), ('BasePriority', SDWORD)]