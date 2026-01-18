import os, sys, threading
import ctypes, msvcrt
from ctypes import POINTER
from ctypes.wintypes import HANDLE, HLOCAL, LPVOID, WORD, DWORD, BOOL, \
def _stderr_raw(self, s):
    """Writes the string to stdout"""
    print(s, end='', file=sys.stderr)
    sys.stderr.flush()