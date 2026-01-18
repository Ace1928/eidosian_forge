import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
@implementer(interfaces.IHandshakeListener)
class CloseAfterHandshake(protocol.Protocol):
    gotData = False

    def __init__(self):
        self.done = defer.Deferred()

    def handshakeCompleted(self):
        self.transport.loseConnection()

    def connectionLost(self, reason):
        self.done.errback(reason)
        del self.done