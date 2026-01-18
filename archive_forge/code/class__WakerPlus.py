from __future__ import annotations
import sys
from zope.interface import implementer
from CFNetwork import (
from CoreFoundation import (
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, PosixReactorBase
from twisted.python import log
from ._signals import _UnixWaker
class _WakerPlus(_UnixWaker):
    """
    The normal Twisted waker will simply wake up the main loop, which causes an
    iteration to run, which in turn causes L{ReactorBase.runUntilCurrent}
    to get invoked.

    L{CFReactor} has a slightly different model of iteration, though: rather
    than have each iteration process the thread queue, then timed calls, then
    file descriptors, each callback is run as it is dispatched by the CFRunLoop
    observer which triggered it.

    So this waker needs to not only unblock the loop, but also make sure the
    work gets done; so, it reschedules the invocation of C{runUntilCurrent} to
    be immediate (0 seconds from now) even if there is no timed call work to
    do.
    """

    def __init__(self, reactor):
        super().__init__()
        self.reactor = reactor

    def doRead(self):
        """
        Wake up the loop and force C{runUntilCurrent} to run immediately in the
        next timed iteration.
        """
        result = super().doRead()
        self.reactor._scheduleSimulate(True)
        return result