from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def __get_stack_trace(self, depth=16, bUseLabels=True, bMakePretty=True):
    """
        Tries to get a stack trace for the current function using the debug
        helper API (dbghelp.dll).

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
    arch = aProcess.get_arch()
    bits = aProcess.get_bits()
    if arch == win32.ARCH_I386:
        MachineType = win32.IMAGE_FILE_MACHINE_I386
    elif arch == win32.ARCH_AMD64:
        MachineType = win32.IMAGE_FILE_MACHINE_AMD64
    elif arch == win32.ARCH_IA64:
        MachineType = win32.IMAGE_FILE_MACHINE_IA64
    else:
        msg = 'Stack walking is not available for this architecture: %s'
        raise NotImplementedError(msg % arch)
    hProcess = aProcess.get_handle(win32.PROCESS_VM_READ | win32.PROCESS_QUERY_INFORMATION)
    hThread = self.get_handle(win32.THREAD_GET_CONTEXT | win32.THREAD_QUERY_INFORMATION)
    StackFrame = win32.STACKFRAME64()
    StackFrame.AddrPC = win32.ADDRESS64(self.get_pc())
    StackFrame.AddrFrame = win32.ADDRESS64(self.get_fp())
    StackFrame.AddrStack = win32.ADDRESS64(self.get_sp())
    trace = list()
    while win32.StackWalk64(MachineType, hProcess, hThread, StackFrame):
        if depth <= 0:
            break
        fp = StackFrame.AddrFrame.Offset
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