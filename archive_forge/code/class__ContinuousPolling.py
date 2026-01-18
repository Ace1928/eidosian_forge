import socket
import sys
from typing import Sequence
from zope.interface import classImplements, implementer
from twisted.internet import error, tcp, udp
from twisted.internet.base import ReactorBase
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform, platformType
from ._signals import (
@implementer(IReactorFDSet)
class _ContinuousPolling(_PollLikeMixin, _DisconnectSelectableMixin):
    """
    Schedule reads and writes based on the passage of time, rather than
    notification.

    This is useful for supporting polling filesystem files, which C{epoll(7)}
    does not support.

    The implementation uses L{_PollLikeMixin}, which is a bit hacky, but
    re-implementing and testing the relevant code yet again is unappealing.

    @ivar _reactor: The L{EPollReactor} that is using this instance.

    @ivar _loop: A C{LoopingCall} that drives the polling, or L{None}.

    @ivar _readers: A C{set} of C{FileDescriptor} objects that should be read
        from.

    @ivar _writers: A C{set} of C{FileDescriptor} objects that should be
        written to.
    """
    _POLL_DISCONNECTED = 1
    _POLL_IN = 2
    _POLL_OUT = 4

    def __init__(self, reactor):
        self._reactor = reactor
        self._loop = None
        self._readers = set()
        self._writers = set()

    def _checkLoop(self):
        """
        Start or stop a C{LoopingCall} based on whether there are readers and
        writers.
        """
        if self._readers or self._writers:
            if self._loop is None:
                from twisted.internet.task import _EPSILON, LoopingCall
                self._loop = LoopingCall(self.iterate)
                self._loop.clock = self._reactor
                self._loop.start(_EPSILON, now=False)
        elif self._loop:
            self._loop.stop()
            self._loop = None

    def iterate(self):
        """
        Call C{doRead} and C{doWrite} on all readers and writers respectively.
        """
        for reader in list(self._readers):
            self._doReadOrWrite(reader, reader, self._POLL_IN)
        for writer in list(self._writers):
            self._doReadOrWrite(writer, writer, self._POLL_OUT)

    def addReader(self, reader):
        """
        Add a C{FileDescriptor} for notification of data available to read.
        """
        self._readers.add(reader)
        self._checkLoop()

    def addWriter(self, writer):
        """
        Add a C{FileDescriptor} for notification of data available to write.
        """
        self._writers.add(writer)
        self._checkLoop()

    def removeReader(self, reader):
        """
        Remove a C{FileDescriptor} from notification of data available to read.
        """
        try:
            self._readers.remove(reader)
        except KeyError:
            return
        self._checkLoop()

    def removeWriter(self, writer):
        """
        Remove a C{FileDescriptor} from notification of data available to
        write.
        """
        try:
            self._writers.remove(writer)
        except KeyError:
            return
        self._checkLoop()

    def removeAll(self):
        """
        Remove all readers and writers.
        """
        result = list(self._readers | self._writers)
        self._readers.clear()
        self._writers.clear()
        return result

    def getReaders(self):
        """
        Return a list of the readers.
        """
        return list(self._readers)

    def getWriters(self):
        """
        Return a list of the writers.
        """
        return list(self._writers)

    def isReading(self, fd):
        """
        Checks if the file descriptor is currently being observed for read
        readiness.

        @param fd: The file descriptor being checked.
        @type fd: L{twisted.internet.abstract.FileDescriptor}
        @return: C{True} if the file descriptor is being observed for read
            readiness, C{False} otherwise.
        @rtype: C{bool}
        """
        return fd in self._readers

    def isWriting(self, fd):
        """
        Checks if the file descriptor is currently being observed for write
        readiness.

        @param fd: The file descriptor being checked.
        @type fd: L{twisted.internet.abstract.FileDescriptor}
        @return: C{True} if the file descriptor is being observed for write
            readiness, C{False} otherwise.
        @rtype: C{bool}
        """
        return fd in self._writers