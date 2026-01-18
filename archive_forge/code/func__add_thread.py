from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def _add_thread(self, aThread):
    """
        Private method to add a thread object to the snapshot.

        @type  aThread: L{Thread}
        @param aThread: Thread object.
        """
    dwThreadId = aThread.dwThreadId
    aThread.set_process(self)
    self.__threadDict[dwThreadId] = aThread