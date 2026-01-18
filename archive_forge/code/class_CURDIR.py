from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class CURDIR(Structure):
    _fields_ = [('DosPath', UNICODE_STRING), ('Handle', PVOID)]