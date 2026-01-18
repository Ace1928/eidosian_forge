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
def _intervalOf(self, t: float) -> int:
    """
        Determine the number of intervals passed as of the given point in
        time.

        @param t: The specified time (from the start of the L{LoopingCall}) to
            be measured in intervals

        @return: The C{int} number of intervals which have passed as of the
            given point in time.
        """
    assert self.starttime is not None
    assert self.interval is not None
    elapsedTime = t - self.starttime
    intervalNum = int(elapsedTime / self.interval)
    return intervalNum