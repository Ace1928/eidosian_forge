from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations
import sys
import ctypes
import warnings
import traceback
class LoadDLLEvent(Event):
    """
    Module load event.
    """
    eventMethod = 'load_dll'
    eventName = 'Module load event'
    eventDescription = 'A new DLL library was loaded by the debugee.'

    def get_module_base(self):
        """
        @rtype:  int
        @return: Base address for the newly loaded DLL.
        """
        return self.raw.u.LoadDll.lpBaseOfDll

    def get_module(self):
        """
        @rtype:  L{Module}
        @return: Module object for the newly loaded DLL.
        """
        lpBaseOfDll = self.get_module_base()
        aProcess = self.get_process()
        if aProcess.has_module(lpBaseOfDll):
            aModule = aProcess.get_module(lpBaseOfDll)
        else:
            aModule = Module(lpBaseOfDll, hFile=self.get_file_handle(), fileName=self.get_filename(), process=aProcess)
            aProcess._add_module(aModule)
        return aModule

    def get_file_handle(self):
        """
        @rtype:  L{FileHandle} or None
        @return: File handle to the newly loaded DLL received from the system.
            Returns C{None} if the handle is not available.
        """
        try:
            hFile = self.__hFile
        except AttributeError:
            hFile = self.raw.u.LoadDll.hFile
            if hFile in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
                hFile = None
            else:
                hFile = FileHandle(hFile, True)
            self.__hFile = hFile
        return hFile

    def get_filename(self):
        """
        @rtype:  str, None
        @return: This method does it's best to retrieve the filename to
        the newly loaded module. However, sometimes that's not
        possible, and C{None} is returned instead.
        """
        szFilename = None
        aProcess = self.get_process()
        lpRemoteFilenamePtr = self.raw.u.LoadDll.lpImageName
        if lpRemoteFilenamePtr:
            lpFilename = aProcess.peek_uint(lpRemoteFilenamePtr)
            fUnicode = bool(self.raw.u.LoadDll.fUnicode)
            szFilename = aProcess.peek_string(lpFilename, fUnicode)
            if not szFilename:
                szFilename = None
        if not szFilename:
            hFile = self.get_file_handle()
            if hFile:
                szFilename = hFile.get_filename()
        return szFilename