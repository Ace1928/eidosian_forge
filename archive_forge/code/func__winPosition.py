import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _winPosition():
    cursor = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor))
    return (cursor.x, cursor.y)