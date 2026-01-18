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
def _setAttributes(self, reactor, done):
    """
        Set attributes on the protocol that are known only externally; this
        will be called by L{runProtocolsWithReactor} when this protocol is
        instantiated.

        @param reactor: The reactor used in this test.

        @param done: A L{Deferred} which will be fired when the connection is
           lost.
        """
    self.reactor = reactor
    self._done = done