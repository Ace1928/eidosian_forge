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
def _serverGetConnectionAddressTest(self, client, interface, which):
    """
        Connect C{client} to a server listening on C{interface} started with
        L{IReactorTCP.listenTCP} and return the address returned by one of the
        server transport's address lookup methods, C{getHost} or C{getPeer}.

        @param client: A C{SOCK_STREAM} L{socket.socket} created with an address
            family such that it will be able to connect to a server listening on
            C{interface}.

        @param interface: A C{str} giving an address for a server to listen on.
            This should almost certainly be the loopback address for some
            address family supported by L{IReactorTCP.listenTCP}.

        @param which: A C{str} equal to either C{"getHost"} or C{"getPeer"}
            determining which address will be returned.

        @return: Whatever object, probably an L{IAddress} provider, is returned
            from the method indicated by C{which}.
        """

    class ObserveAddress(Protocol):

        def makeConnection(self, transport):
            reactor.stop()
            self.factory.address = getattr(transport, which)()
    reactor = self.buildReactor()
    factory = ServerFactory()
    factory.protocol = ObserveAddress
    port = self.getListeningPort(reactor, factory, 0, interface)
    client.setblocking(False)
    try:
        connect(client, (port.getHost().host, port.getHost().port))
    except OSError as e:
        self.assertIn(e.errno, (errno.EINPROGRESS, errno.EWOULDBLOCK))
    self.runReactor(reactor)
    return factory.address