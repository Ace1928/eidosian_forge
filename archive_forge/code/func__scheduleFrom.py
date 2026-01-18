import sys
import time
import warnings
from typing import (
from zope.interface import implementer
from incremental import Version
from twisted.internet.base import DelayedCall
from twisted.internet.defer import Deferred, ensureDeferred, maybeDeferred
from twisted.internet.error import ReactorNotRunning
from twisted.internet.interfaces import IDelayedCall, IReactorCore, IReactorTime
from twisted.python import log, reflect
from twisted.python.deprecate import _getDeprecationWarningString
from twisted.python.failure import Failure
def _scheduleFrom(self, when: float) -> None:
    """
        Schedule the next iteration of this looping call.

        @param when: The present time from whence the call is scheduled.
        """

    def howLong() -> float:
        if self.interval == 0:
            return 0
        assert self.starttime is not None
        runningFor = when - self.starttime
        assert self.interval is not None
        untilNextInterval = self.interval - runningFor % self.interval
        if when == when + untilNextInterval:
            return self.interval
        return untilNextInterval
    self.call = self.clock.callLater(howLong(), self)