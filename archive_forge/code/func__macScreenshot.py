import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _macScreenshot(filename=None):
    if filename is not None:
        tmpFilename = filename
    else:
        tmpFilename = 'screenshot%s.png' % datetime.datetime.now().strftime('%Y-%m%d_%H-%M-%S-%f')
    subprocess.call(['screencapture', '-x', tmpFilename])
    im = Image.open(tmpFilename)
    im.load()
    if filename is None:
        os.unlink(tmpFilename)
    return im