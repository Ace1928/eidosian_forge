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
def add_existing_session(self, dwProcessId, bStarted=False):
    """
        Use this method only when for some reason the debugger's been attached
        to the target outside of WinAppDbg (for example when integrating with
        other tools).

        You don't normally need to call this method. Most users should call
        L{attach}, L{execv} or L{execl} instead.

        @type  dwProcessId: int
        @param dwProcessId: Global process ID.

        @type  bStarted: bool
        @param bStarted: C{True} if the process was started by the debugger,
            or C{False} if the process was attached to instead.

        @raise WindowsError: The target process does not exist, is not attached
            to the debugger anymore.
        """
    if not self.system.has_process(dwProcessId):
        aProcess = Process(dwProcessId)
        self.system._add_process(aProcess)
    else:
        aProcess = self.system.get_process(dwProcessId)
    aProcess.get_handle()
    if bStarted:
        self.__attachedDebugees.add(dwProcessId)
    else:
        self.__startedDebugees.add(dwProcessId)
    self.__setSystemKillOnExitMode()
    aProcess.scan_threads()
    aProcess.scan_modules()