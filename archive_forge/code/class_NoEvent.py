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
class NoEvent(Event):
    """
    No event.

    Dummy L{Event} object that can be used as a placeholder when no debug
    event has occured yet. It's never returned by the L{EventFactory}.
    """
    eventMethod = 'no_event'
    eventName = 'No event'
    eventDescription = 'No debug event has occured.'

    def __init__(self, debug, raw=None):
        Event.__init__(self, debug, raw)

    def __len__(self):
        """
        Always returns C{0}, so when evaluating the object as a boolean it's
        always C{False}. This prevents L{Debug.cont} from trying to continue
        a dummy event.
        """
        return 0

    def get_event_code(self):
        return -1

    def get_pid(self):
        return -1

    def get_tid(self):
        return -1

    def get_process(self):
        return Process(self.get_pid())

    def get_thread(self):
        return Thread(self.get_tid())