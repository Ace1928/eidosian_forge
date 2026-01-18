from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __postCallAction_codebp(self, event):
    """
        Handles code breakpoint events on return from the function.

        @type  event: L{ExceptionEvent}
        @param event: Breakpoint hit event.
        """
    tid = event.get_tid()
    if tid not in self.__paramStack:
        return True
    pid = event.get_pid()
    address = event.breakpoint.get_address()
    event.debug.dont_break_at(pid, address)
    try:
        self.__postCallAction(event)
    finally:
        self.__pop_params(tid)