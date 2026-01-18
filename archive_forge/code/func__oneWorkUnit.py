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
def _oneWorkUnit(self) -> None:
    """
        Perform one unit of work for this task, retrieving one item from its
        iterator, stopping if there are no further items in the iterator, and
        pausing if the result was a L{Deferred}.
        """
    try:
        result = next(self._iterator)
    except StopIteration:
        self._completeWith(TaskDone(), self._iterator)
    except BaseException:
        self._completeWith(TaskFailed(), Failure())
    else:
        if isinstance(result, Deferred):
            self.pause()

            def failLater(failure: Failure) -> None:
                self._completeWith(TaskFailed(), failure)
            result.addCallbacks(lambda result: self.resume(), failLater)