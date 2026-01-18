from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
class SYSDBG_MSR(Structure):
    _fields_ = [('Address', ULONG), ('Data', ULONGLONG)]