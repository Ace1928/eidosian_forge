from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __get_running_bp_set(self, tid):
    """Auxiliary method."""
    return self.__runningBP.get(tid, ())