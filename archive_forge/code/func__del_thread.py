from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def _del_thread(self, dwThreadId):
    """
        Private method to remove a thread object from the snapshot.

        @type  dwThreadId: int
        @param dwThreadId: Global thread ID.
        """
    try:
        aThread = self.__threadDict[dwThreadId]
        del self.__threadDict[dwThreadId]
    except KeyError:
        aThread = None
        msg = 'Unknown thread ID %d' % dwThreadId
        warnings.warn(msg, RuntimeWarning)
    if aThread:
        aThread.clear()