import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _linuxGetPixel(x, y):
    rgbValue = screenshot().getpixel((x, y))
    return (rgbValue[0], rgbValue[1], rgbValue[2])