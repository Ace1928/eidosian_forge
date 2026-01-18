import os, sys, threading
import ctypes, msvcrt
from ctypes import POINTER
from ctypes.wintypes import HANDLE, HLOCAL, LPVOID, WORD, DWORD, BOOL, \
def _stdout_thread(self, handle, func):
    data = ctypes.create_string_buffer(4096)
    while True:
        bytesRead = DWORD(0)
        if not ReadFile(handle, data, 4096, ctypes.byref(bytesRead), None):
            le = GetLastError()
            if le == ERROR_BROKEN_PIPE:
                return
            else:
                raise ctypes.WinError()
        s = data.value[0:bytesRead.value]
        func(s.decode('utf_8', 'replace'))