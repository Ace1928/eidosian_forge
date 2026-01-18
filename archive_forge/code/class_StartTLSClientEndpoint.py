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
@implementer(IStreamClientEndpoint)
class StartTLSClientEndpoint:
    """
    An endpoint which wraps another one and adds a TLS layer immediately when
    connections are set up.

    @ivar wrapped: A L{IStreamClientEndpoint} provider which will be used to
        really set up connections.

    @ivar contextFactory: A L{ContextFactory} to use to do TLS.
    """

    def __init__(self, wrapped, contextFactory):
        self.wrapped = wrapped
        self.contextFactory = contextFactory

    def connect(self, factory):
        """
        Establish a connection using a protocol build by C{factory} and
        immediately start TLS on it.  Return a L{Deferred} which fires with the
        protocol instance.
        """

        class WrapperFactory(ServerFactory):

            def buildProtocol(wrapperSelf, addr):
                protocol = factory.buildProtocol(addr)

                def connectionMade(orig=protocol.connectionMade):
                    protocol.transport.startTLS(self.contextFactory)
                    orig()
                protocol.connectionMade = connectionMade
                return protocol
        return self.wrapped.connect(WrapperFactory())