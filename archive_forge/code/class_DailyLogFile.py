import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable
class DailyLogFile(BaseLogFile):
    """A log file that is rotated daily (at or after midnight localtime)"""

    def _openFile(self):
        BaseLogFile._openFile(self)
        self.lastDate = self.toDate(os.stat(self.path)[8])

    def shouldRotate(self):
        """Rotate when the date has changed since last write"""
        return self.toDate() > self.lastDate

    def toDate(self, *args):
        """Convert a unixtime to (year, month, day) localtime tuple,
        or return the current (year, month, day) localtime tuple.

        This function primarily exists so you may overload it with
        gmtime, or some cruft to make unit testing possible.
        """
        return time.localtime(*args)[:3]

    def suffix(self, tupledate):
        """Return the suffix given a (year, month, day) tuple or unixtime"""
        try:
            return '_'.join(map(str, tupledate))
        except BaseException:
            return '_'.join(map(str, self.toDate(tupledate)))

    def getLog(self, identifier):
        """Given a unix time, return a LogReader for an old log file."""
        if self.toDate(identifier) == self.lastDate:
            return self.getCurrentLog()
        filename = f'{self.path}.{self.suffix(identifier)}'
        if not os.path.exists(filename):
            raise ValueError('no such logfile exists')
        return LogReader(filename)

    def write(self, data):
        """Write some data to the log file"""
        BaseLogFile.write(self, data)
        self.lastDate = max(self.lastDate, self.toDate())

    def rotate(self):
        """Rotate the file and create a new one.

        If it's not possible to open new logfile, this will fail silently,
        and continue logging to old logfile.
        """
        if not (os.access(self.directory, os.W_OK) and os.access(self.path, os.W_OK)):
            return
        newpath = f'{self.path}.{self.suffix(self.lastDate)}'
        if os.path.exists(newpath):
            return
        self._file.close()
        os.rename(self.path, newpath)
        self._openFile()

    def __getstate__(self):
        state = BaseLogFile.__getstate__(self)
        del state['lastDate']
        return state