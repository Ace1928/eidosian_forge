from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __pop_params(self, tid):
    """
        Forgets the arguments tuple for the last call to the hooked function
        from this thread.

        @type  tid: int
        @param tid: Thread global ID.
        """
    stack = self.__paramStack[tid]
    stack.pop()
    if not stack:
        del self.__paramStack[tid]