from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def __get_pid_by_scanning(self):
    """Internally used by get_pid()."""
    dwProcessId = None
    dwThreadId = self.get_tid()
    with win32.CreateToolhelp32Snapshot(win32.TH32CS_SNAPTHREAD) as hSnapshot:
        te = win32.Thread32First(hSnapshot)
        while te is not None:
            if te.th32ThreadID == dwThreadId:
                dwProcessId = te.th32OwnerProcessID
                break
            te = win32.Thread32Next(hSnapshot)
    if dwProcessId is None:
        msg = 'Cannot find thread ID %d in any process' % dwThreadId
        raise RuntimeError(msg)
    return dwProcessId