import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class ProcessInformation(object):
    """
    Process information object returned by L{CreateProcess}.
    """

    def __init__(self, pi):
        self.hProcess = ProcessHandle(pi.hProcess)
        self.hThread = ThreadHandle(pi.hThread)
        self.dwProcessId = pi.dwProcessId
        self.dwThreadId = pi.dwThreadId