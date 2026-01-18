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
class TCPClientTestsBase(ReactorBuilder, ConnectionTestsMixin, StreamClientTestsMixin):
    """
    Base class for builders defining tests related to
    L{IReactorTCP.connectTCP}.  Classes which uses this in must provide all of
    the documented instance variables in order to specify how the test works.
    These are documented as instance variables rather than declared as methods
    due to some peculiar inheritance ordering concerns, but they are
    effectively abstract methods.

    @ivar endpoints: A client/server endpoint creator appropriate to the
        address family being tested.
    @type endpoints: L{twisted.internet.test.connectionmixins.EndpointCreator}

    @ivar interface: An IP address literal to locally bind a socket to as well
        as to connect to.  This can be any valid interface for the local host.
    @type interface: C{str}

    @ivar port: An unused local listening port to listen on and connect to.
        This will be used in conjunction with the C{interface}.  (Depending on
        what they're testing, some tests will locate their own port with
        L{findFreePort} instead.)
    @type port: C{int}

    @ivar family: an address family constant, such as L{socket.AF_INET},
        L{socket.AF_INET6}, or L{socket.AF_UNIX}, which indicates the address
        family of the transport type under test.
    @type family: C{int}

    @ivar addressClass: the L{twisted.internet.interfaces.IAddress} implementor
        associated with the transport type under test.  Must also be a
        3-argument callable which produces an instance of same.
    @type addressClass: C{type}

    @ivar fakeDomainName: A fake domain name to use, to simulate hostname
        resolution and to distinguish between hostnames and IP addresses where
        necessary.
    @type fakeDomainName: C{str}
    """
    requiredInterfaces = (IReactorTCP,)
    _port = None

    @property
    def port(self):
        """
        Return the port number to connect to, using C{self._port} set up by
        C{listen} if available.

        @return: The port number to connect to.
        @rtype: C{int}
        """
        if self._port is not None:
            return self._port.getHost().port
        return findFreePort(self.interface, self.family)[1]

    @property
    def interface(self):
        """
        Return the interface attribute from the endpoints object.
        """
        return self.endpoints.interface

    def listen(self, reactor, factory):
        """
        Start a TCP server with the given C{factory}.

        @param reactor: The reactor to create the TCP port in.

        @param factory: The server factory.

        @return: A TCP port instance.
        """
        self._port = reactor.listenTCP(0, factory, interface=self.interface)
        return self._port

    def connect(self, reactor, factory):
        """
        Start a TCP client with the given C{factory}.

        @param reactor: The reactor to create the connection in.

        @param factory: The client factory.

        @return: A TCP connector instance.
        """
        return reactor.connectTCP(self.interface, self.port, factory)

    def test_buildProtocolReturnsNone(self):
        """
        When the factory's C{buildProtocol} returns L{None} the connection is
        gracefully closed.
        """
        connectionLost = Deferred()
        reactor = self.buildReactor()
        serverFactory = MyServerFactory()
        serverFactory.protocolConnectionLost = connectionLost
        stopOnError(self, reactor)

        class NoneFactory(ServerFactory):

            def buildProtocol(self, address):
                return None
        listening = self.endpoints.server(reactor).listen(serverFactory)

        def listened(port):
            clientFactory = NoneFactory()
            endpoint = self.endpoints.client(reactor, port.getHost())
            return endpoint.connect(clientFactory)
        connecting = listening.addCallback(listened)

        def connectSucceeded(protocol):
            self.fail('Stream client endpoint connect succeeded with %r, should have failed with NoProtocol.' % (protocol,))

        def connectFailed(reason):
            reason.trap(NoProtocol)
        connecting.addCallbacks(connectSucceeded, connectFailed)

        def connected(ignored):
            return connectionLost
        disconnecting = connecting.addCallback(connected)
        disconnecting.addErrback(log.err)

        def disconnected(ignored):
            reactor.stop()
        disconnecting.addCallback(disconnected)
        self.runReactor(reactor)

    def test_addresses(self):
        """
        A client's transport's C{getHost} and C{getPeer} return L{IPv4Address}
        instances which have the dotted-quad string form of the resolved
        address of the local and remote endpoints of the connection
        respectively as their C{host} attribute, not the hostname originally
        passed in to
        L{connectTCP<twisted.internet.interfaces.IReactorTCP.connectTCP>}, if a
        hostname was used.
        """
        host, ignored = findFreePort(self.interface, self.family)[:2]
        reactor = self.buildReactor()
        fakeDomain = self.fakeDomainName
        reactor.installResolver(FakeResolver({fakeDomain: self.interface}))
        server = reactor.listenTCP(0, ServerFactory.forProtocol(Protocol), interface=host)
        serverAddress = server.getHost()
        transportData = {'host': None, 'peer': None, 'instance': None}

        class CheckAddress(Protocol):

            def makeConnection(self, transport):
                transportData['host'] = transport.getHost()
                transportData['peer'] = transport.getPeer()
                transportData['instance'] = transport
                reactor.stop()
        clientFactory = Stop(reactor)
        clientFactory.protocol = CheckAddress

        def connectMe():
            while True:
                port = findFreePort(self.interface, self.family)
                bindAddress = (self.interface, port[1])
                log.msg(f'Connect attempt with bindAddress {bindAddress}')
                try:
                    reactor.connectTCP(fakeDomain, server.getHost().port, clientFactory, bindAddress=bindAddress)
                except ConnectBindError:
                    continue
                else:
                    clientFactory.boundPort = port[1]
                    break
        needsRunningReactor(reactor, connectMe)
        self.runReactor(reactor)
        if clientFactory.failReason:
            self.fail(clientFactory.failReason.getTraceback())
        transportRepr = '<{} to {} at {:x}>'.format(transportData['instance'].__class__, transportData['instance'].addr, id(transportData['instance']))
        boundPort = [host] + list(socket.getaddrinfo(self.interface, clientFactory.boundPort)[0][-1][1:])
        serverPort = [host] + list(socket.getaddrinfo(self.interface, serverAddress.port)[0][-1][1:])
        self.assertEqual(transportData['host'], self.addressClass('TCP', *boundPort))
        self.assertEqual(transportData['peer'], self.addressClass('TCP', *serverPort))
        self.assertEqual(repr(transportData['instance']), transportRepr)

    def test_badContext(self):
        """
        If the context factory passed to L{ITCPTransport.startTLS} raises an
        exception from its C{getContext} method, that exception is raised by
        L{ITCPTransport.startTLS}.
        """
        reactor = self.buildReactor()
        brokenFactory = BrokenContextFactory()
        results = []
        serverFactory = ServerFactory.forProtocol(Protocol)
        port = reactor.listenTCP(0, serverFactory, interface=self.interface)
        endpoint = self.endpoints.client(reactor, port.getHost())
        clientFactory = ClientFactory()
        clientFactory.protocol = Protocol
        connectDeferred = endpoint.connect(clientFactory)

        def connected(protocol):
            if not ITLSTransport.providedBy(protocol.transport):
                results.append('skip')
            else:
                results.append(self.assertRaises(ValueError, protocol.transport.startTLS, brokenFactory))

        def connectFailed(failure):
            results.append(failure)

        def whenRun():
            connectDeferred.addCallback(connected)
            connectDeferred.addErrback(connectFailed)
            connectDeferred.addBoth(lambda ign: reactor.stop())
        needsRunningReactor(reactor, whenRun)
        self.runReactor(reactor)
        self.assertEqual(len(results), 1, f'more than one callback result: {results}')
        if isinstance(results[0], Failure):
            results[0].raiseException()
        if results[0] == 'skip':
            raise SkipTest('Reactor does not support ITLSTransport')
        self.assertEqual(BrokenContextFactory.message, str(results[0]))