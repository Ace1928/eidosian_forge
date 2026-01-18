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
class CreateProcessEvent(Event):
    """
    Process creation event.
    """
    eventMethod = 'create_process'
    eventName = 'Process creation event'
    eventDescription = 'A new process has started.'

    def get_file_handle(self):
        """
        @rtype:  L{FileHandle} or None
        @return: File handle to the main module, received from the system.
            Returns C{None} if the handle is not available.
        """
        try:
            hFile = self.__hFile
        except AttributeError:
            hFile = self.raw.u.CreateProcessInfo.hFile
            if hFile in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
                hFile = None
            else:
                hFile = FileHandle(hFile, True)
            self.__hFile = hFile
        return hFile

    def get_process_handle(self):
        """
        @rtype:  L{ProcessHandle}
        @return: Process handle received from the system.
            Returns C{None} if the handle is not available.
        """
        hProcess = self.raw.u.CreateProcessInfo.hProcess
        if hProcess in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
            hProcess = None
        else:
            hProcess = ProcessHandle(hProcess, False, win32.PROCESS_ALL_ACCESS)
        return hProcess

    def get_thread_handle(self):
        """
        @rtype:  L{ThreadHandle}
        @return: Thread handle received from the system.
            Returns C{None} if the handle is not available.
        """
        hThread = self.raw.u.CreateProcessInfo.hThread
        if hThread in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
            hThread = None
        else:
            hThread = ThreadHandle(hThread, False, win32.THREAD_ALL_ACCESS)
        return hThread

    def get_start_address(self):
        """
        @rtype:  int
        @return: Pointer to the first instruction to execute in this process.

            Returns C{NULL} when the debugger attaches to a process.

            See U{http://msdn.microsoft.com/en-us/library/ms679295(VS.85).aspx}
        """
        return self.raw.u.CreateProcessInfo.lpStartAddress

    def get_image_base(self):
        """
        @rtype:  int
        @return: Base address of the main module.
        @warn: This value is taken from the PE file
            and may be incorrect because of ASLR!
        """
        return self.raw.u.CreateProcessInfo.lpBaseOfImage

    def get_teb(self):
        """
        @rtype:  int
        @return: Pointer to the TEB.
        """
        return self.raw.u.CreateProcessInfo.lpThreadLocalBase

    def get_debug_info(self):
        """
        @rtype:  str
        @return: Debugging information.
        """
        raw = self.raw.u.CreateProcessInfo
        ptr = raw.lpBaseOfImage + raw.dwDebugInfoFileOffset
        size = raw.nDebugInfoSize
        data = self.get_process().peek(ptr, size)
        if len(data) == size:
            return data
        return None

    def get_filename(self):
        """
        @rtype:  str, None
        @return: This method does it's best to retrieve the filename to
        the main module of the process. However, sometimes that's not
        possible, and C{None} is returned instead.
        """
        szFilename = None
        hFile = self.get_file_handle()
        if hFile:
            szFilename = hFile.get_filename()
        if not szFilename:
            aProcess = self.get_process()
            lpRemoteFilenamePtr = self.raw.u.CreateProcessInfo.lpImageName
            if lpRemoteFilenamePtr:
                lpFilename = aProcess.peek_uint(lpRemoteFilenamePtr)
                fUnicode = bool(self.raw.u.CreateProcessInfo.fUnicode)
                szFilename = aProcess.peek_string(lpFilename, fUnicode)
            if not szFilename:
                szFilename = aProcess.get_image_name()
        return szFilename

    def get_module_base(self):
        """
        @rtype:  int
        @return: Base address of the main module.
        """
        return self.get_image_base()

    def get_module(self):
        """
        @rtype:  L{Module}
        @return: Main module of the process.
        """
        return self.get_process().get_module(self.get_module_base())