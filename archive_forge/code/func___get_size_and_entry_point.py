from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def __get_size_and_entry_point(self):
    """Get the size and entry point of the module using the Win32 API."""
    process = self.get_process()
    if process:
        try:
            handle = process.get_handle(win32.PROCESS_VM_READ | win32.PROCESS_QUERY_INFORMATION)
            base = self.get_base()
            mi = win32.GetModuleInformation(handle, base)
            self.SizeOfImage = mi.SizeOfImage
            self.EntryPoint = mi.EntryPoint
        except WindowsError:
            e = sys.exc_info()[1]
            warnings.warn('Cannot get size and entry point of module %s, reason: %s' % (self.get_name(), e.strerror), RuntimeWarning)