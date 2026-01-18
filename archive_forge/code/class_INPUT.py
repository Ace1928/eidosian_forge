import ctypes
import ctypes.wintypes
import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT
import sys
class INPUT(ctypes.Structure):

    class _I(ctypes.Union):
        _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT), ('hi', HARDWAREINPUT)]
    _anonymous_ = ('i',)
    _fields_ = [('type', ctypes.wintypes.DWORD), ('i', _I)]