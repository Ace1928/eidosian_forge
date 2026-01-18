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
def __find_processes_by_filename(self, filename):
    """
        Internally used by L{find_processes_by_filename}.
        """
    found = list()
    filename = filename.lower()
    if PathOperations.path_is_absolute(filename):
        for aProcess in self.iter_processes():
            imagename = aProcess.get_filename()
            if imagename and imagename.lower() == filename:
                found.append((aProcess, imagename))
    else:
        for aProcess in self.iter_processes():
            imagename = aProcess.get_filename()
            if imagename:
                imagename = PathOperations.pathname_to_filename(imagename)
                if imagename.lower() == filename:
                    found.append((aProcess, imagename))
    return found