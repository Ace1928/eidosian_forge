import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _copyRgbHexMouseInfo(self, *args):
    if len(args) > 0 and isinstance(args[0], Event):
        args = ()
    if self.delayEnabledSV.get() == 'on' and len(args) == 0:
        self.root.after(1000, self._copyRgbHexMouseInfo, 2)
        self.rgbHexCopyButtonSV.set('Copy in 3')
    elif len(args) == 1 and args[0] == 2:
        self.root.after(1000, self._copyRgbHexMouseInfo, 1)
        self.rgbHexCopyButtonSV.set('Copy in 2')
    elif len(args) == 1 and args[0] == 1:
        self.root.after(1000, self._copyRgbHexMouseInfo, 0)
        self.rgbHexCopyButtonSV.set('Copy in 1')
    else:
        self._copyText(self.rgbHexSV.get())
        self.rgbHexCopyButtonSV.set('Copy RGB Hex')