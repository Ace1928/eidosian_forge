import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
class TimeoutReportReactor(PosixReactorBase):
    """
    A reactor which is just barely runnable and which cannot monitor any
    readers or writers, and which fires a L{Deferred} with the timeout
    passed to its C{doIteration} method as soon as that method is invoked.
    """

    def __init__(self):
        PosixReactorBase.__init__(self)
        self.iterationTimeout = Deferred()
        self.now = 100

    def addReader(self, reader: IReadDescriptor) -> None:
        """
        Ignore the reader.  This is necessary because the waker will be
        added.  However, we won't actually monitor it for any events.
        """

    def removeReader(self, reader: IReadDescriptor) -> None:
        """
        See L{addReader}.
        """

    def removeAll(self):
        """
        There are no readers or writers, so there is nothing to remove.
        This will be called when the reactor stops, though, so it must be
        implemented.
        """
        return []

    def seconds(self):
        """
        Override the real clock with a deterministic one that can be easily
        controlled in a unit test.
        """
        return self.now

    def doIteration(self, timeout):
        d = self.iterationTimeout
        if d is not None:
            self.iterationTimeout = None
            d.callback(timeout)