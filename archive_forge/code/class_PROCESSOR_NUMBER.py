from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class PROCESSOR_NUMBER(Structure):
    _fields_ = [('Group', WORD), ('Number', BYTE), ('Reserved', BYTE)]