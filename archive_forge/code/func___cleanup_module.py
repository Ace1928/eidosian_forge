from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __cleanup_module(self, event):
    """
        Auxiliary method for L{_notify_unload_dll}.
        """
    pid = event.get_pid()
    process = event.get_process()
    module = event.get_module()
    for tid in process.iter_thread_ids():
        thread = process.get_thread(tid)
        if tid in self.__runningBP:
            bplist = list(self.__runningBP[tid])
            for bp in bplist:
                bp_address = bp.get_address()
                if process.get_module_at_address(bp_address) == module:
                    self.__cleanup_breakpoint(event, bp)
                    self.__runningBP[tid].remove(bp)
        if tid in self.__hardwareBP:
            bplist = list(self.__hardwareBP[tid])
            for bp in bplist:
                bp_address = bp.get_address()
                if process.get_module_at_address(bp_address) == module:
                    self.__cleanup_breakpoint(event, bp)
                    self.__hardwareBP[tid].remove(bp)
    for bp_pid, bp_address in compat.keys(self.__codeBP):
        if bp_pid == pid:
            if process.get_module_at_address(bp_address) == module:
                bp = self.__codeBP[bp_pid, bp_address]
                self.__cleanup_breakpoint(event, bp)
                del self.__codeBP[bp_pid, bp_address]
    for bp_pid, bp_address in compat.keys(self.__pageBP):
        if bp_pid == pid:
            if process.get_module_at_address(bp_address) == module:
                bp = self.__pageBP[bp_pid, bp_address]
                self.__cleanup_breakpoint(event, bp)
                del self.__pageBP[bp_pid, bp_address]