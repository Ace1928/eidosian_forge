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
def _addTask(self, task: CooperativeTask) -> None:
    """
        Add a L{CooperativeTask} object to this L{Cooperator}.
        """
    if self._stopped:
        self._tasks.append(task)
        task._completeWith(SchedulerStopped(), Failure(SchedulerStopped()))
    else:
        self._tasks.append(task)
        self._reschedule()