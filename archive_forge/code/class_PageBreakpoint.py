from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
class PageBreakpoint(Breakpoint):
    """
    Page access breakpoint (using guard pages).

    @see: L{Debug.watch_buffer}

    @group Information:
        get_size_in_pages
    """
    typeName = 'page breakpoint'

    def __init__(self, address, pages=1, condition=True, action=None):
        """
        Page breakpoint object.

        @see: L{Breakpoint.__init__}

        @type  address: int
        @param address: Memory address for breakpoint.

        @type  pages: int
        @param address: Size of breakpoint in pages.

        @type  condition: function
        @param condition: (Optional) Condition callback function.

        @type  action: function
        @param action: (Optional) Action callback function.
        """
        Breakpoint.__init__(self, address, pages * MemoryAddresses.pageSize, condition, action)
        floordiv_align = long(address) // long(MemoryAddresses.pageSize)
        truediv_align = float(address) / float(MemoryAddresses.pageSize)
        if floordiv_align != truediv_align:
            msg = 'Address of page breakpoint must be aligned to a page size boundary (value %s received)' % HexDump.address(address)
            raise ValueError(msg)

    def get_size_in_pages(self):
        """
        @rtype:  int
        @return: The size in pages of the breakpoint.
        """
        return self.get_size() // MemoryAddresses.pageSize

    def __set_bp(self, aProcess):
        """
        Sets the target pages as guard pages.

        @type  aProcess: L{Process}
        @param aProcess: Process object.
        """
        lpAddress = self.get_address()
        dwSize = self.get_size()
        flNewProtect = aProcess.mquery(lpAddress).Protect
        flNewProtect = flNewProtect | win32.PAGE_GUARD
        aProcess.mprotect(lpAddress, dwSize, flNewProtect)

    def __clear_bp(self, aProcess):
        """
        Restores the original permissions of the target pages.

        @type  aProcess: L{Process}
        @param aProcess: Process object.
        """
        lpAddress = self.get_address()
        flNewProtect = aProcess.mquery(lpAddress).Protect
        flNewProtect = flNewProtect & (4294967295 ^ win32.PAGE_GUARD)
        aProcess.mprotect(lpAddress, self.get_size(), flNewProtect)

    def disable(self, aProcess, aThread):
        if not self.is_disabled():
            self.__clear_bp(aProcess)
        super(PageBreakpoint, self).disable(aProcess, aThread)

    def enable(self, aProcess, aThread):
        if win32.arch not in (win32.ARCH_I386, win32.ARCH_AMD64):
            msg = 'Only one-shot page breakpoints are supported for %s'
            raise NotImplementedError(msg % win32.arch)
        if not self.is_enabled() and (not self.is_one_shot()):
            self.__set_bp(aProcess)
        super(PageBreakpoint, self).enable(aProcess, aThread)

    def one_shot(self, aProcess, aThread):
        if not self.is_enabled() and (not self.is_one_shot()):
            self.__set_bp(aProcess)
        super(PageBreakpoint, self).one_shot(aProcess, aThread)

    def running(self, aProcess, aThread):
        aThread.set_tf()
        super(PageBreakpoint, self).running(aProcess, aThread)