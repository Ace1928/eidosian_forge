from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def __add_loaded_module(self, event):
    """
        Private method to automatically add new module objects from debug events.

        @type  event: L{Event}
        @param event: Event object.
        """
    lpBaseOfDll = event.get_module_base()
    hFile = event.get_file_handle()
    if lpBaseOfDll not in self.__moduleDict:
        fileName = event.get_filename()
        if not fileName:
            fileName = None
        if hasattr(event, 'get_start_address'):
            EntryPoint = event.get_start_address()
        else:
            EntryPoint = None
        aModule = Module(lpBaseOfDll, hFile, fileName=fileName, EntryPoint=EntryPoint, process=self)
        self._add_module(aModule)
    else:
        aModule = self.get_module(lpBaseOfDll)
        if not aModule.hFile and hFile not in (None, 0, win32.INVALID_HANDLE_VALUE):
            aModule.hFile = hFile
        if not aModule.process:
            aModule.process = self
        if aModule.EntryPoint is None and hasattr(event, 'get_start_address'):
            aModule.EntryPoint = event.get_start_address()
        if not aModule.fileName:
            fileName = event.get_filename()
            if fileName:
                aModule.fileName = fileName