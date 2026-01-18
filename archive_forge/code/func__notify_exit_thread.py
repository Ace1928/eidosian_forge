from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def _notify_exit_thread(self, event):
    """
        Notify the termination of a thread.

        @type  event: L{ExitThreadEvent}
        @param event: Exit thread event.

        @rtype:  bool
        @return: C{True} to call the user-defined handler, C{False} otherwise.
        """
    self.__cleanup_thread(event)
    return True