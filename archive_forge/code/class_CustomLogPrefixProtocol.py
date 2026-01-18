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
class CustomLogPrefixProtocol(ConnectableProtocol):

    def __init__(self, prefix):
        self._prefix = prefix
        self.system = None

    def connectionMade(self):
        self.transport.write(b'a')

    def logPrefix(self):
        return self._prefix

    def dataReceived(self, bytes):
        self.system = context.get(ILogContext)['system']
        self.transport.write(b'b')
        if b'b' in bytes:
            self.transport.loseConnection()