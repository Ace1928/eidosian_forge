import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable
class LogReader:
    """Read from a log file."""

    def __init__(self, name):
        """
        Open the log file for reading.

        The comments about binary-mode for L{BaseLogFile._openFile} also apply
        here.
        """
        self._file = open(name)

    def readLines(self, lines=10):
        """Read a list of lines from the log file.

        This doesn't returns all of the files lines - call it multiple times.
        """
        result = []
        for i in range(lines):
            line = self._file.readline()
            if not line:
                break
            result.append(line)
        return result

    def close(self):
        self._file.close()