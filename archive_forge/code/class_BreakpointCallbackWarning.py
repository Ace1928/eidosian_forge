from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
class BreakpointCallbackWarning(RuntimeWarning):
    """
    This warning is issued when an uncaught exception was raised by a
    breakpoint's user-defined callback.
    """