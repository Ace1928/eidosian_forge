import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _winGetPixel(x, y):
    colorRef = ctypes.windll.gdi32.GetPixel(dc, x, y)
    red = colorRef % 256
    colorRef //= 256
    green = colorRef % 256
    colorRef //= 256
    blue = colorRef
    return (red, green, blue)