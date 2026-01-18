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
class RecordingClientProtocol(protocol.Protocol):
    """
    @ivar deferred: a deferred that will fire with first received content.
    @type deferred: L{defer.Deferred}
    """

    def __init__(self):
        self.deferred = defer.Deferred()

    def connectionMade(self):
        self.transport.getPeerCertificate()

    def dataReceived(self, data):
        self.deferred.callback(data)