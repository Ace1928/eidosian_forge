from __future__ import with_statement
from winappdbg.textio import HexDump
from winappdbg import win32
import ctypes
import warnings
class DistormEngine(Engine):
    """
    Integration with the diStorm disassembler by Gil Dabah.

    @see: U{https://code.google.com/p/distorm3}
    """
    name = 'diStorm'
    desc = 'diStorm disassembler by Gil Dabah'
    url = 'https://code.google.com/p/distorm3'
    supported = set((win32.ARCH_I386, win32.ARCH_AMD64))

    def _import_dependencies(self):
        global distorm3
        if distorm3 is None:
            try:
                import distorm3
            except ImportError:
                import distorm as distorm3
        self.__decode = distorm3.Decode
        self.__flag = {win32.ARCH_I386: distorm3.Decode32Bits, win32.ARCH_AMD64: distorm3.Decode64Bits}[self.arch]

    def decode(self, address, code):
        return self.__decode(address, code, self.__flag)