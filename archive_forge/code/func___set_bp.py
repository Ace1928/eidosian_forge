from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __set_bp(self, aThread):
    """
        Sets this breakpoint in the debug registers.

        @type  aThread: L{Thread}
        @param aThread: Thread object.
        """
    if self.__slot is None:
        aThread.suspend()
        try:
            ctx = aThread.get_context(win32.CONTEXT_DEBUG_REGISTERS)
            self.__slot = DebugRegister.find_slot(ctx)
            if self.__slot is None:
                msg = 'No available hardware breakpoint slots for thread ID %d'
                msg = msg % aThread.get_tid()
                raise RuntimeError(msg)
            DebugRegister.set_bp(ctx, self.__slot, self.get_address(), self.__trigger, self.__watch)
            aThread.set_context(ctx)
        finally:
            aThread.resume()