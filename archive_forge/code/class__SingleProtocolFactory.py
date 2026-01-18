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
class _SingleProtocolFactory(ClientFactory):
    """
    Factory to be used by L{runProtocolsWithReactor}.

    It always returns the same protocol (i.e. is intended for only a single
    connection).
    """

    def __init__(self, protocol):
        self._protocol = protocol

    def buildProtocol(self, addr):
        return self._protocol