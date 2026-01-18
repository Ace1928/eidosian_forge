from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class RTL_ACTIVATION_CONTEXT_STACK_FRAME(Structure):
    _fields_ = [('Previous', PVOID), ('ActivationContext', PVOID), ('Flags', DWORD)]