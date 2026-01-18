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
class AdoptStreamConnectionTestsBuilder(TCPTransportServerAddressTestMixin, WriteSequenceTestsMixin, ReactorBuilder):
    """
    Test server transports built using C{adoptStreamConnection}.
    """
    requiredInterfaces = (IReactorFDSet, IReactorSocket)

    def getConnectedClientAndServer(self, reactor, interface, addressFamily):
        """
        Return a L{Deferred} firing with a L{MyClientFactory} and
        L{MyServerFactory} connected pair, and the listening C{Port}. The
        particularity is that the server protocol has been obtained after doing
        a C{adoptStreamConnection} against the original server connection.
        """
        firstServer = MyServerFactory()
        firstServer.protocolConnectionMade = Deferred()
        server = MyServerFactory()
        server.protocolConnectionMade = Deferred()
        server.protocolConnectionLost = Deferred()
        client = MyClientFactory()
        client.protocolConnectionMade = Deferred()
        client.protocolConnectionLost = Deferred()
        port = reactor.listenTCP(0, firstServer, interface=interface)

        def firtServerConnected(proto):
            reactor.removeReader(proto.transport)
            reactor.removeWriter(proto.transport)
            reactor.adoptStreamConnection(proto.transport.fileno(), addressFamily, server)
        firstServer.protocolConnectionMade.addCallback(firtServerConnected)
        lostDeferred = gatherResults([client.protocolConnectionLost, server.protocolConnectionLost])

        def stop(result):
            if reactor.running:
                reactor.stop()
            return result
        lostDeferred.addBoth(stop)
        deferred = Deferred()
        deferred.addErrback(stop)
        startDeferred = gatherResults([client.protocolConnectionMade, server.protocolConnectionMade])

        def start(protocols):
            client, server = protocols
            log.msg('client connected %s' % client)
            log.msg('server connected %s' % server)
            deferred.callback((client, server, port))
        startDeferred.addCallback(start)
        reactor.connectTCP(interface, port.getHost().port, client)
        return deferred