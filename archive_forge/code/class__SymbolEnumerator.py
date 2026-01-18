from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
class _SymbolEnumerator(object):
    """
        Internally used by L{Module} to enumerate symbols in a module.
        """

    def __init__(self, undecorate=False):
        self.symbols = list()
        self.undecorate = undecorate

    def __call__(self, SymbolName, SymbolAddress, SymbolSize, UserContext):
        """
            Callback that receives symbols and stores them in a Python list.
            """
        if self.undecorate:
            try:
                SymbolName = win32.UnDecorateSymbolName(SymbolName)
            except Exception:
                pass
        self.symbols.append((SymbolName, SymbolAddress, SymbolSize))
        return win32.TRUE