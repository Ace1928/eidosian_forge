import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _macPosition():
    loc = NSEvent.mouseLocation
    return (int(loc.x), int(core_graphics.CGDisplayPixelsHigh(0) - loc.y))