import socket
from gc import collect
from typing import Optional
from weakref import ref
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.interfaces import IConnector, IReactorFDSet
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.reactormixins import needsRunningReactor
from twisted.python import context, log
from twisted.python.failure import Failure
from twisted.python.log import ILogContext, err, msg
from twisted.python.runtime import platform
from twisted.test.test_tcp import ClosingProtocol
from twisted.trial.unittest import SkipTest
class _AcceptOneClient(ServerFactory):
    """
    This factory fires a L{Deferred} with a protocol instance shortly after it
    is constructed (hopefully long enough afterwards so that it has been
    connected to a transport).

    @ivar reactor: The reactor used to schedule the I{shortly}.

    @ivar result: A L{Deferred} which will be fired with the protocol instance.
    """

    def __init__(self, reactor, result):
        self.reactor = reactor
        self.result = result

    def buildProtocol(self, addr):
        protocol = ServerFactory.buildProtocol(self, addr)
        self.reactor.callLater(0, self.result.callback, protocol)
        return protocol