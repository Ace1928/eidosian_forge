import ctypes
import ctypes.wintypes
import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT
import sys
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [('dx', ctypes.wintypes.LONG), ('dy', ctypes.wintypes.LONG), ('mouseData', ctypes.wintypes.DWORD), ('dwFlags', ctypes.wintypes.DWORD), ('time', ctypes.wintypes.DWORD), ('dwExtraInfo', ctypes.POINTER(ctypes.wintypes.ULONG))]