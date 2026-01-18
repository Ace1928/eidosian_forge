from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __clear_break(self, pid, address):
    """
        Used by L{dont_break_at} and L{dont_stalk_at}.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.
        """
    if type(address) not in (int, long):
        unknown = True
        label = address
        try:
            deferred = self.__deferredBP[pid]
            del deferred[label]
            unknown = False
        except KeyError:
            pass
        aProcess = self.system.get_process(pid)
        try:
            address = aProcess.resolve_label(label)
            if not address:
                raise Exception()
        except Exception:
            if unknown:
                msg = "Can't clear unknown code breakpoint at %s in process ID %d"
                msg = msg % (label, pid)
                warnings.warn(msg, BreakpointWarning)
            return
    if self.has_code_breakpoint(pid, address):
        self.erase_code_breakpoint(pid, address)