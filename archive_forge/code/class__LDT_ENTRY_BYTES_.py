from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_I386
class _LDT_ENTRY_BYTES_(Structure):
    _pack_ = 1
    _fields_ = [('BaseMid', BYTE), ('Flags1', BYTE), ('Flags2', BYTE), ('BaseHi', BYTE)]