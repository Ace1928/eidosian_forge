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
def __get_hooks_for_dll(self, event):
    """
        Get the requested API hooks for the current DLL.

        Used by L{__hook_dll} and L{__unhook_dll}.
        """
    result = []
    if self.__apiHooks:
        path = event.get_module().get_filename()
        if path:
            lib_name = PathOperations.pathname_to_filename(path).lower()
            for hook_lib, hook_api_list in compat.iteritems(self.__apiHooks):
                if hook_lib == lib_name:
                    result.extend(hook_api_list)
    return result