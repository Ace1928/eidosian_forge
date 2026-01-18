import os, sys, threading
import ctypes, msvcrt
from ctypes import POINTER
from ctypes.wintypes import HANDLE, HLOCAL, LPVOID, WORD, DWORD, BOOL, \
def _stdin_raw_block(self):
    """Use a blocking stdin read"""
    try:
        data = sys.stdin.read(1)
        data = data.replace('\r', '\n')
        return data
    except WindowsError as we:
        if we.winerror == ERROR_NO_DATA:
            return None
        else:
            raise we