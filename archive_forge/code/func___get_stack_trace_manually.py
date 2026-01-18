from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def __get_stack_trace_manually(self, depth=16, bUseLabels=True, bMakePretty=True):
    """
        Tries to get a stack trace for the current function.
        Only works for functions with standard prologue and epilogue.

        @type  depth: int
        @param depth: Maximum depth of stack trace.

        @type  bUseLabels: bool
        @param bUseLabels: C{True} to use labels, C{False} to use addresses.

        @type  bMakePretty: bool
        @param bMakePretty:
            C{True} for user readable labels,
            C{False} for labels that can be passed to L{Process.resolve_label}.

            "Pretty" labels look better when producing output for the user to
            read, while pure labels are more useful programatically.

        @rtype:  tuple of tuple( int, int, str )
        @return: Stack trace of the thread as a tuple of
            ( return address, frame pointer address, module filename )
            when C{bUseLabels} is C{True}, or a tuple of
            ( return address, frame pointer label )
            when C{bUseLabels} is C{False}.

        @raise WindowsError: Raises an exception on error.
        """
    aProcess = self.get_process()
    st, sb = self.get_stack_range()
    fp = self.get_fp()
    trace = list()
    if aProcess.get_module_count() == 0:
        aProcess.scan_modules()
    bits = aProcess.get_bits()
    while depth > 0:
        if fp == 0:
            break
        if not st <= fp < sb:
            break
        ra = aProcess.peek_pointer(fp + 4)
        if ra == 0:
            break
        lib = aProcess.get_module_at_address(ra)
        if lib is None:
            lib = ''
        elif lib.fileName:
            lib = lib.fileName
        else:
            lib = '%s' % HexDump.address(lib.lpBaseOfDll, bits)
        if bUseLabels:
            label = aProcess.get_label_at_address(ra)
            if bMakePretty:
                label = '%s (%s)' % (HexDump.address(ra, bits), label)
            trace.append((fp, label))
        else:
            trace.append((fp, ra, lib))
        fp = aProcess.peek_pointer(fp)
    return tuple(trace)