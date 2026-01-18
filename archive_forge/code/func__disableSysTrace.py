import sys, re, traceback, threading
from .. import exceptionHandling as exceptionHandling
from ..Qt import QtWidgets, QtCore
from ..functions import SignalBlock
from .stackwidget import StackWidget
def _disableSysTrace(self):
    sys.settrace(None)
    threading.settrace(None)
    if hasattr(threading, 'settrace_all_threads'):
        threading.settrace_all_threads(None)