from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __stop_tracing(self, thread):
    """
        @type  thread: L{Thread}
        @param thread: Thread to stop tracing.
        """
    tid = thread.get_tid()
    if tid in self.__tracing:
        self.__tracing.remove(tid)
        if thread.is_alive():
            thread.clear_tf()