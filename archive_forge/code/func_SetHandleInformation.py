import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetHandleInformation(hObject, dwMask, dwFlags):
    _SetHandleInformation = windll.kernel32.SetHandleInformation
    _SetHandleInformation.argtypes = [HANDLE, DWORD, DWORD]
    _SetHandleInformation.restype = bool
    _SetHandleInformation.errcheck = RaiseIfZero
    _SetHandleInformation(hObject, dwMask, dwFlags)