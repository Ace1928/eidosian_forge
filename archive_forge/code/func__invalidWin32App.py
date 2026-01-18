import os
import sys
from zope.interface import implementer
import pywintypes
import win32api
import win32con
import win32event
import win32file
import win32pipe
import win32process
import win32security
from twisted.internet import _pollingfile, error
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IConsumer, IProcessTransport, IProducer
from twisted.python.win32 import quoteArguments
def _invalidWin32App(pywinerr):
    """
    Determine if a pywintypes.error is telling us that the given process is
    'not a valid win32 application', i.e. not a PE format executable.

    @param pywinerr: a pywintypes.error instance raised by CreateProcess

    @return: a boolean
    """
    return pywinerr.args[0] == 193