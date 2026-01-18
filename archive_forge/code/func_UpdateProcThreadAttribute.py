import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def UpdateProcThreadAttribute(lpAttributeList, Attribute, Value, cbSize=None):
    _UpdateProcThreadAttribute = windll.kernel32.UpdateProcThreadAttribute
    _UpdateProcThreadAttribute.argtypes = [LPPROC_THREAD_ATTRIBUTE_LIST, DWORD, DWORD_PTR, PVOID, SIZE_T, PVOID, PSIZE_T]
    _UpdateProcThreadAttribute.restype = bool
    _UpdateProcThreadAttribute.errcheck = RaiseIfZero
    if cbSize is None:
        cbSize = sizeof(Value)
    _UpdateProcThreadAttribute(byref(lpAttributeList), 0, Attribute, byref(Value), cbSize, None, None)