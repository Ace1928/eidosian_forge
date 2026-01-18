import os
import struct
from twisted.internet import fdesc
from twisted.internet.abstract import FileDescriptor
from twisted.python import _inotify, log
def _rmWatch(self, wd):
    """
        Private helper that abstracts the use of ctypes.

        Calls the internal inotify API to remove an fd from inotify then
        removes the corresponding watchpoint from the internal mapping together
        with the file descriptor from the watchpath.
        """
    self._inotify.remove(self._fd, wd)
    iwp = self._watchpoints.pop(wd)
    self._watchpaths.pop(iwp.path)