from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __callHandler(self, callback, event, *params):
    """
        Calls a "pre" or "post" handler, if set.

        @type  callback: function
        @param callback: Callback function to call.

        @type  event: L{ExceptionEvent}
        @param event: Breakpoint hit event.

        @type  params: tuple
        @param params: Parameters for the callback function.
        """
    if callback is not None:
        event.hook = self
        callback(event, *params)