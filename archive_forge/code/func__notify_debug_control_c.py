import sys
from winappdbg import win32
from winappdbg.system import System
from winappdbg.process import Process
from winappdbg.thread import Thread
from winappdbg.module import Module
from winappdbg.window import Window
from winappdbg.breakpoint import _BreakpointContainer, CodeBreakpoint
from winappdbg.event import Event, EventHandler, EventDispatcher, EventFactory
from winappdbg.interactive import ConsoleDebugger
import warnings
def _notify_debug_control_c(self, event):
    """
        Notify of a Debug Ctrl-C exception.

        @warning: This method is meant to be used internally by the debugger.

        @note: This exception is only raised when a debugger is attached, and
            applications are not supposed to handle it, so we need to handle it
            ourselves or the application may crash.

        @see: U{http://msdn.microsoft.com/en-us/library/aa363082(VS.85).aspx}

        @type  event: L{ExceptionEvent}
        @param event: Debug Ctrl-C exception event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
    if event.is_first_chance():
        event.continueStatus = win32.DBG_EXCEPTION_HANDLED
    return True