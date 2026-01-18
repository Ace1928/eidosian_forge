from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def _notify_guard_page(self, event):
    """
        Notify breakpoints of a guard page exception event.

        @type  event: L{ExceptionEvent}
        @param event: Guard page exception event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
    address = event.get_fault_address()
    pid = event.get_pid()
    bCallHandler = True
    mask = ~(MemoryAddresses.pageSize - 1)
    address = address & mask
    key = (pid, address)
    if key in self.__pageBP:
        bp = self.__pageBP[key]
        if bp.is_enabled() or bp.is_one_shot():
            event.continueStatus = win32.DBG_CONTINUE
            bp.hit(event)
            if bp.is_running():
                tid = event.get_tid()
                self.__add_running_bp(tid, bp)
            bCondition = bp.eval_condition(event)
            if bCondition and bp.is_automatic():
                bp.run_action(event)
                bCallHandler = False
            else:
                bCallHandler = bCondition
    else:
        event.continueStatus = win32.DBG_EXCEPTION_NOT_HANDLED
    return bCallHandler