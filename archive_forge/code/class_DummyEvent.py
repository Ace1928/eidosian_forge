from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.util import PathOperations
from winappdbg.event import EventHandler, NoEvent
from winappdbg.textio import HexInput, HexOutput, HexDump, CrashDump, DebugLog
import os
import sys
import code
import time
import warnings
import traceback
from cmd import Cmd
class DummyEvent(NoEvent):
    """Dummy event object used internally by L{ConsoleDebugger}."""

    def get_pid(self):
        return self._pid

    def get_tid(self):
        return self._tid

    def get_process(self):
        return self._process

    def get_thread(self):
        return self._thread