from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations
import sys
import ctypes
import warnings
import traceback
class ExitThreadEvent(Event):
    """
    Thread termination event.
    """
    eventMethod = 'exit_thread'
    eventName = 'Thread termination event'
    eventDescription = 'A thread has finished executing.'

    def get_exit_code(self):
        """
        @rtype:  int
        @return: Exit code of the thread.
        """
        return self.raw.u.ExitThread.dwExitCode