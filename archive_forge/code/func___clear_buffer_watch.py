from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __clear_buffer_watch(self, bw):
    """
        Used by L{dont_watch_buffer} and L{dont_stalk_buffer}.

        @type  bw: L{BufferWatch}
        @param bw: Buffer watch identifier.
        """
    pid = bw.pid
    start = bw.start
    end = bw.end
    base = MemoryAddresses.align_address_to_page_start(start)
    limit = MemoryAddresses.align_address_to_page_end(end)
    pages = MemoryAddresses.get_buffer_size_in_pages(start, end - start)
    cset = set()
    page_addr = base
    pageSize = MemoryAddresses.pageSize
    while page_addr < limit:
        if self.has_page_breakpoint(pid, page_addr):
            bp = self.get_page_breakpoint(pid, page_addr)
            condition = bp.get_condition()
            if condition not in cset:
                if not isinstance(condition, _BufferWatchCondition):
                    continue
                cset.add(condition)
                condition.remove(bw)
                if condition.count() == 0:
                    try:
                        self.erase_page_breakpoint(pid, bp.get_address())
                    except WindowsError:
                        msg = 'Cannot remove page breakpoint at address %s'
                        msg = msg % HexDump.address(bp.get_address())
                        warnings.warn(msg, BreakpointWarning)
        page_addr = page_addr + pageSize