from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __cleanup_process(self, event):
    """
        Auxiliary method for L{_notify_exit_process}.
        """
    pid = event.get_pid()
    process = event.get_process()
    for bp_pid, bp_address in compat.keys(self.__codeBP):
        if bp_pid == pid:
            bp = self.__codeBP[bp_pid, bp_address]
            self.__cleanup_breakpoint(event, bp)
            del self.__codeBP[bp_pid, bp_address]
    for bp_pid, bp_address in compat.keys(self.__pageBP):
        if bp_pid == pid:
            bp = self.__pageBP[bp_pid, bp_address]
            self.__cleanup_breakpoint(event, bp)
            del self.__pageBP[bp_pid, bp_address]
    try:
        del self.__deferredBP[pid]
    except KeyError:
        pass