import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _logRgbHexMouseInfo(self, *args):
    if len(args) > 0 and isinstance(args[0], Event):
        args = ()
    if self.delayEnabledSV.get() == 'on' and len(args) == 0:
        self.root.after(1000, self._logRgbHexMouseInfo, 2)
        self.rgbHexLogButtonSV.set('Log in 3')
    elif len(args) == 1 and args[0] == 2:
        self.root.after(1000, self._logRgbHexMouseInfo, 1)
        self.rgbHexLogButtonSV.set('Log in 2')
    elif len(args) == 1 and args[0] == 1:
        self.root.after(1000, self._logRgbHexMouseInfo, 0)
        self.rgbHexLogButtonSV.set('Log in 1')
    else:
        logContents = self.logTextarea.get('1.0', 'end-1c') + '%s\n' % self.rgbHexSV.get()
        self.logTextboxSV.set(logContents)
        self._setLogTextAreaContents(logContents)
        self.statusbarSV.set('Logged ' + self.rgbHexSV.get())
        self.rgbHexLogButtonSV.set('Log RGB Hex')