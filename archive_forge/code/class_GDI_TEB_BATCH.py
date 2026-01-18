from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class GDI_TEB_BATCH(Structure):
    _fields_ = [('Offset', ULONG), ('HDC', ULONG), ('Buffer', ULONG * 310)]