from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump, HexInput
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses
from winappdbg.module import Module, _ModuleContainer
from winappdbg.thread import Thread, _ThreadContainer
from winappdbg.window import Window
from winappdbg.search import Search, \
from winappdbg.disasm import Disassembler
import re
import os
import os.path
import ctypes
import struct
import warnings
import traceback
def __fixup_labels(self, disasm):
    """
        Private method used when disassembling from process memory.

        It has no return value because the list is modified in place. On return
        all raw memory addresses are replaced by labels when possible.

        @type  disasm: list of tuple(int, int, str, str)
        @param disasm: Output of one of the dissassembly functions.
        """
    for index in compat.xrange(len(disasm)):
        address, size, text, dump = disasm[index]
        m = self.__hexa_parameter.search(text)
        while m:
            s, e = m.span()
            value = text[s:e]
            try:
                label = self.get_label_at_address(int(value, 16))
            except Exception:
                label = None
            if label:
                text = text[:s] + label + text[e:]
                e = s + len(value)
            m = self.__hexa_parameter.search(text, e)
        disasm[index] = (address, size, text, dump)