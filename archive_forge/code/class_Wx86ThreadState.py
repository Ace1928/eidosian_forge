from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class Wx86ThreadState(Structure):
    _fields_ = [('CallBx86Eip', PVOID), ('DeallocationCpu', PVOID), ('UseKnownWx86Dll', UCHAR), ('OleStubInvoked', CHAR)]