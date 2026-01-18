import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
class TCPServerTests(TestCase):
    """
    Whitebox tests for L{twisted.internet.tcp.Server}.
    """

    def setUp(self):
        self.reactor = _FakeFDSetReactor()

        class FakePort:
            _realPortNumber = 3
        self.skt = FakeSocket(b'')
        self.protocol = Protocol()
        self.server = Server(self.skt, self.protocol, ('', 0), FakePort(), None, self.reactor)

    def test_writeAfterDisconnect(self):
        """
        L{Server.write} discards bytes passed to it if called after it has lost
        its connection.
        """
        self.server.connectionLost(Failure(Exception('Simulated lost connection')))
        self.server.write(b'hello world')
        self.assertEqual(self.skt.sendBuffer, [])

    def test_writeAfterDisconnectAfterTLS(self):
        """
        L{Server.write} discards bytes passed to it if called after it has lost
        its connection when the connection had started TLS.
        """
        self.server.TLS = True
        self.test_writeAfterDisconnect()

    def test_writeSequenceAfterDisconnect(self):
        """
        L{Server.writeSequence} discards bytes passed to it if called after it
        has lost its connection.
        """
        self.server.connectionLost(Failure(Exception('Simulated lost connection')))
        self.server.writeSequence([b'hello world'])
        self.assertEqual(self.skt.sendBuffer, [])

    def test_writeSequenceAfterDisconnectAfterTLS(self):
        """
        L{Server.writeSequence} discards bytes passed to it if called after it
        has lost its connection when the connection had started TLS.
        """
        self.server.TLS = True
        self.test_writeSequenceAfterDisconnect()