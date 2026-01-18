from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
class RiggedDailyLogFile(logfile.DailyLogFile):
    _clock = 0.0

    def _openFile(self) -> None:
        logfile.DailyLogFile._openFile(self)
        self.lastDate = self.toDate()

    def toDate(self, *args: float) -> tuple[int, int, int]:
        if args:
            return time.gmtime(*args)[:3]
        return time.gmtime(self._clock)[:3]