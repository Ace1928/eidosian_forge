from __future__ import annotations
from io import BytesIO
from socket import AF_INET, AF_INET6
from typing import Callable, Iterator, Sequence, overload
from zope.interface import implementedBy, implementer
from zope.interface.verify import verifyClass
from typing_extensions import ParamSpec, Self
from twisted.internet import address, error, protocol, task
from twisted.internet.abstract import _dataMustBeBytes, isIPv6Address
from twisted.internet.address import IPv4Address, IPv6Address, UNIXAddress
from twisted.internet.defer import Deferred
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import (
from twisted.internet.task import Clock
from twisted.logger import ILogObserver, LogEvent, LogPublisher
from twisted.protocols import basic
from twisted.python import failure
from twisted.trial.unittest import TestCase
@implementer(ILogObserver)
class EventLoggingObserver(Sequence[LogEvent]):
    """
    L{ILogObserver} That stores its events in a list for later inspection.
    This class is similar to L{LimitedHistoryLogObserver} save that the
    internal buffer is public and intended for external inspection.  The
    observer implements the sequence protocol to ease iteration of the events.

    @ivar _events: The events captured by this observer
    @type _events: L{list}
    """

    def __init__(self) -> None:
        self._events: list[LogEvent] = []

    def __len__(self) -> int:
        return len(self._events)

    @overload
    def __getitem__(self, index: int) -> LogEvent:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[LogEvent]:
        ...

    def __getitem__(self, index: int | slice) -> LogEvent | Sequence[LogEvent]:
        return self._events[index]

    def __iter__(self) -> Iterator[LogEvent]:
        return iter(self._events)

    def __call__(self, event: LogEvent) -> None:
        """
        @see: L{ILogObserver}
        """
        self._events.append(event)

    @classmethod
    def createWithCleanup(cls, testInstance: TestCase, publisher: LogPublisher) -> Self:
        """
        Create an L{EventLoggingObserver} instance that observes the provided
        publisher and will be cleaned up with addCleanup().

        @param testInstance: Test instance in which this logger is used.
        @type testInstance: L{twisted.trial.unittest.TestCase}

        @param publisher: Log publisher to observe.
        @type publisher: twisted.logger.LogPublisher

        @return: An EventLoggingObserver configured to observe the provided
            publisher.
        @rtype: L{twisted.test.proto_helpers.EventLoggingObserver}
        """
        obs = cls()
        publisher.addObserver(obs)
        testInstance.addCleanup(lambda: publisher.removeObserver(obs))
        return obs