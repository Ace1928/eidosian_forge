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
class FakeSocketTests(TestCase):
    """
    Test that the FakeSocket can be used by the doRead method of L{Connection}
    """

    def test_blocking(self):
        skt = FakeSocket(b'someData')
        skt.setblocking(0)
        self.assertEqual(skt.blocking, 0)

    def test_recv(self):
        skt = FakeSocket(b'someData')
        self.assertEqual(skt.recv(10), b'someData')

    def test_send(self):
        """
        L{FakeSocket.send} accepts the entire string passed to it, adds it to
        its send buffer, and returns its length.
        """
        skt = FakeSocket(b'')
        count = skt.send(b'foo')
        self.assertEqual(count, 3)
        self.assertEqual(skt.sendBuffer, [b'foo'])