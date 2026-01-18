import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _linuxPosition():
    coord = _display.screen().root.query_pointer()._data
    return (coord['root_x'], coord['root_y'])