from typing import Optional, Sequence, Type
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.endpoints import (
from twisted.internet.error import ConnectionClosed
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
class TLSPortTestsBuilder(TLSMixin, ContextGeneratingMixin, BadContextTestsMixin, ConnectToTCPListenerMixin, StreamTransportTestsMixin, ReactorBuilder):
    """
    Tests for L{IReactorSSL.listenSSL}
    """

    def getListeningPort(self, reactor, factory):
        """
        Get a TLS port from a reactor.
        """
        return reactor.listenSSL(0, factory, self.getServerContext())

    def getExpectedStartListeningLogMessage(self, port, factory):
        """
        Get the message expected to be logged when a TLS port starts listening.
        """
        return '%s (TLS) starting on %d' % (factory, port.getHost().port)

    def getExpectedConnectionLostLogMsg(self, port):
        """
        Get the expected connection lost message for a TLS port.
        """
        return f'(TLS Port {port.getHost().port} Closed)'

    def test_badContext(self):
        """
        If the context factory passed to L{IReactorSSL.listenSSL} raises an
        exception from its C{getContext} method, that exception is raised by
        L{IReactorSSL.listenSSL}.
        """

        def useIt(reactor, contextFactory):
            return reactor.listenSSL(0, ServerFactory(), contextFactory)
        self._testBadContext(useIt)

    def connectToListener(self, reactor, address, factory):
        """
        Connect to the given listening TLS port, assuming the
        underlying transport is TCP.

        @param reactor: The reactor under test.
        @type reactor: L{IReactorSSL}

        @param address: The listening's address.  Only the C{port}
            component is used; see
            L{ConnectToTCPListenerMixin.LISTENER_HOST}.
        @type address: L{IPv4Address} or L{IPv6Address}

        @param factory: The client factory.
        @type factory: L{ClientFactory}

        @return: The connector
        """
        return reactor.connectSSL(self.LISTENER_HOST, address.port, factory, self.getClientContext())