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
class CreateThreadEvent(Event):
    """
    Thread creation event.
    """
    eventMethod = 'create_thread'
    eventName = 'Thread creation event'
    eventDescription = 'A new thread has started.'

    def get_thread_handle(self):
        """
        @rtype:  L{ThreadHandle}
        @return: Thread handle received from the system.
            Returns C{None} if the handle is not available.
        """
        hThread = self.raw.u.CreateThread.hThread
        if hThread in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
            hThread = None
        else:
            hThread = ThreadHandle(hThread, False, win32.THREAD_ALL_ACCESS)
        return hThread

    def get_teb(self):
        """
        @rtype:  int
        @return: Pointer to the TEB.
        """
        return self.raw.u.CreateThread.lpThreadLocalBase

    def get_start_address(self):
        """
        @rtype:  int
        @return: Pointer to the first instruction to execute in this thread.

            Returns C{NULL} when the debugger attached to a process
            and the thread already existed.

            See U{http://msdn.microsoft.com/en-us/library/ms679295(VS.85).aspx}
        """
        return self.raw.u.CreateThread.lpStartAddress