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
class _Reaper(_pollingfile._PollableResource):

    def __init__(self, proc):
        self.proc = proc

    def checkWork(self):
        if win32event.WaitForSingleObject(self.proc.hProcess, 0) != win32event.WAIT_OBJECT_0:
            return 0
        exitCode = win32process.GetExitCodeProcess(self.proc.hProcess)
        self.deactivate()
        self.proc.processEnded(exitCode)
        return 0