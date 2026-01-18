import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetHandleInformation(hObject):
    _GetHandleInformation = windll.kernel32.GetHandleInformation
    _GetHandleInformation.argtypes = [HANDLE, PDWORD]
    _GetHandleInformation.restype = bool
    _GetHandleInformation.errcheck = RaiseIfZero
    dwFlags = DWORD(0)
    _GetHandleInformation(hObject, byref(dwFlags))
    return dwFlags.value