import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
class UnreadingWriter(protocol.Protocol):
    """
            Trivial protocol which pauses its transport immediately and then
            writes some bytes to it.
            """

    def connectionMade(self):
        msg('UnreadingWriter.connectionMade')
        self.transport.pauseProducing()
        clientPaused.callback(None)
        msg('clientPaused called back')

        def write(ignored):
            msg('UnreadingWriter.connectionMade write')
            producer = Infinite(self.transport)
            msg('UnreadingWriter.connectionMade write created producer')
            self.transport.registerProducer(producer, False)
            msg('UnreadingWriter.connectionMade write registered producer')
        serverLost.addCallback(write)