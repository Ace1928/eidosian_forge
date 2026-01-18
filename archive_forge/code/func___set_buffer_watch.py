from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __set_buffer_watch(self, pid, address, size, action, bOneShot):
    """
        Used by L{watch_buffer} and L{stalk_buffer}.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int
        @param address: Memory address of buffer to watch.

        @type  size: int
        @param size: Size in bytes of buffer to watch.

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_page_breakpoint} for more details.

        @type  bOneShot: bool
        @param bOneShot:
            C{True} to set a one-shot breakpoint,
            C{False} to set a normal breakpoint.
        """
    if size < 1:
        raise ValueError('Bad size for buffer watch: %r' % size)
    bw = BufferWatch(pid, address, address + size, action, bOneShot)
    base = MemoryAddresses.align_address_to_page_start(address)
    limit = MemoryAddresses.align_address_to_page_end(address + size)
    pages = MemoryAddresses.get_buffer_size_in_pages(address, size)
    try:
        bset = set()
        nset = set()
        cset = set()
        page_addr = base
        pageSize = MemoryAddresses.pageSize
        while page_addr < limit:
            if self.has_page_breakpoint(pid, page_addr):
                bp = self.get_page_breakpoint(pid, page_addr)
                if bp not in bset:
                    condition = bp.get_condition()
                    if not condition in cset:
                        if not isinstance(condition, _BufferWatchCondition):
                            msg = "Can't watch buffer at page %s"
                            msg = msg % HexDump.address(page_addr)
                            raise RuntimeError(msg)
                        cset.add(condition)
                    bset.add(bp)
            else:
                condition = _BufferWatchCondition()
                bp = self.define_page_breakpoint(pid, page_addr, 1, condition=condition)
                bset.add(bp)
                nset.add(bp)
                cset.add(condition)
            page_addr = page_addr + pageSize
        aProcess = self.system.get_process(pid)
        for bp in bset:
            if bp.is_disabled() or bp.is_one_shot():
                bp.enable(aProcess, None)
    except:
        for bp in nset:
            try:
                self.erase_page_breakpoint(pid, bp.get_address())
            except:
                pass
        raise
    for condition in cset:
        condition.add(bw)