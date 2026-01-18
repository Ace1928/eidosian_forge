from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
class AccumulatingProtocol(Protocol):
    """
    A protocol which collects the bytes it receives and closes its connection
    after receiving a certain minimum of data.

    @ivar howMany: The number of bytes of data to wait for before closing the
        connection.

    @ivar received: A L{list} of L{bytes} of the bytes received so far.
    """

    def __init__(self, howMany):
        self.howMany = howMany

    def connectionMade(self):
        self.received = []

    def dataReceived(self, data):
        self.received.append(data)
        if sum(map(len, self.received)) >= self.howMany:
            self.transport.loseConnection()

    def connectionLost(self, reason):
        if not reason.check(ConnectionDone):
            log.err(reason)