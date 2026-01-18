import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def DeviceIoControl(hDevice, dwIoControlCode, lpInBuffer, nInBufferSize, lpOutBuffer, nOutBufferSize, lpOverlapped):
    _DeviceIoControl = windll.kernel32.DeviceIoControl
    _DeviceIoControl.argtypes = [HANDLE, DWORD, LPVOID, DWORD, LPVOID, DWORD, LPDWORD, LPOVERLAPPED]
    _DeviceIoControl.restype = bool
    _DeviceIoControl.errcheck = RaiseIfZero
    if not lpInBuffer:
        lpInBuffer = None
    if not lpOutBuffer:
        lpOutBuffer = None
    if lpOverlapped:
        lpOverlapped = ctypes.pointer(lpOverlapped)
    lpBytesReturned = DWORD(0)
    _DeviceIoControl(hDevice, dwIoControlCode, lpInBuffer, nInBufferSize, lpOutBuffer, nOutBufferSize, byref(lpBytesReturned), lpOverlapped)
    return lpBytesReturned.value