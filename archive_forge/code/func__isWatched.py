import os
import struct
from twisted.internet import fdesc
from twisted.internet.abstract import FileDescriptor
from twisted.python import _inotify, log
def _isWatched(self, path):
    """
        Helper function that checks if the path is already monitored
        and returns its watchdescriptor if so or None otherwise.

        @param path: The path that should be checked
        @type path: L{FilePath}
        """
    path = path.asBytesMode()
    return self._watchpaths.get(path, None)