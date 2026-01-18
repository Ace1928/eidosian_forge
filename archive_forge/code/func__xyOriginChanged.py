import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _xyOriginChanged(self, sv):
    contents = sv.get()
    if len(contents.split(',')) != 2:
        return
    x, y = contents.split(',')
    x = x.strip()
    y = y.strip()
    if not x.isdecimal() or not y.isdecimal():
        return
    self.xOrigin = int(x)
    self.yOrigin = int(y)
    self.statusbarSV.set('Set XY Origin to ' + str(self.xOrigin) + ', ' + str(self.yOrigin))