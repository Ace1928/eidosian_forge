import sys, os, unicodedata
import py
from py.builtin import text, bytes
def GetConsoleInfo(handle):
    info = CONSOLE_SCREEN_BUFFER_INFO()
    _GetConsoleScreenBufferInfo(handle, ctypes.byref(info))
    return info