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
def __setSystemKillOnExitMode(self):
    if self.__firstDebugee:
        try:
            System.set_kill_on_exit_mode(self.__bKillOnExit)
            self.__firstDebugee = False
        except Exception:
            pass