from __future__ import with_statement
from winappdbg.textio import HexDump
from winappdbg import win32
import ctypes
import warnings
def _validate_arch(self, arch=None):
    """
        @type  arch: str
        @param arch: Name of the processor architecture.
            If not provided the current processor architecture is assumed.
            For more details see L{win32.version._get_arch}.

        @rtype:  str
        @return: Name of the processor architecture.
            If not provided the current processor architecture is assumed.
            For more details see L{win32.version._get_arch}.

        @raise NotImplementedError: This disassembler doesn't support the
            requested processor architecture.
        """
    if not arch:
        arch = win32.arch
    if arch not in self.supported:
        msg = 'The %s engine cannot decode %s code.'
        msg = msg % (self.name, arch)
        raise NotImplementedError(msg)
    return arch