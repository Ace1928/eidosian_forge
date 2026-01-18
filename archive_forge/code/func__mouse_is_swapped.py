import ctypes
import ctypes.wintypes
import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT
import sys
def _mouse_is_swapped():
    return ctypes.windll.user32.GetSystemMetrics(23) != 0