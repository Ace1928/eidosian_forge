import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _setLogTextAreaContents(self, logContents):
    if RUNNING_PYTHON_2:
        self.logTextarea.delete('1.0', tkinter.END)
        self.logTextarea.insert(tkinter.END, logContents)
    else:
        self.logTextarea.replace('1.0', tkinter.END, logContents)
    topOfTextArea, bottomOfTextArea = self.logTextarea.yview()
    self.logTextarea.yview_moveto(bottomOfTextArea)