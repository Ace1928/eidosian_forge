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
class EndpointCreator:
    """
    Create client and server endpoints that know how to connect to each other.
    """

    def server(self, reactor):
        """
        Return an object providing C{IStreamServerEndpoint} for use in creating
        a server to use to establish the connection type to be tested.
        """
        raise NotImplementedError()

    def client(self, reactor, serverAddress):
        """
        Return an object providing C{IStreamClientEndpoint} for use in creating
        a client to use to establish the connection type to be tested.
        """
        raise NotImplementedError()