import ctypes
import ctypes.wintypes
import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT
import sys
class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [('uMsg', ctypes.wintypes.DWORD), ('wParamL', ctypes.wintypes.WORD), ('wParamH', ctypes.wintypes.DWORD)]