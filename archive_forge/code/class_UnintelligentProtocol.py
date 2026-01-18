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
class UnintelligentProtocol(basic.LineReceiver):
    """
    @ivar deferred: a deferred that will fire at connection lost.
    @type deferred: L{defer.Deferred}

    @cvar pretext: text sent before TLS is set up.
    @type pretext: C{bytes}

    @cvar posttext: text sent after TLS is set up.
    @type posttext: C{bytes}
    """
    pretext = [b'first line', b'last thing before tls starts', b'STARTTLS']
    posttext = [b'first thing after tls started', b'last thing ever']

    def __init__(self):
        self.deferred = defer.Deferred()

    def connectionMade(self):
        for l in self.pretext:
            self.sendLine(l)

    def lineReceived(self, line):
        if line == b'READY':
            self.transport.startTLS(ClientTLSContext(), self.factory.client)
            for l in self.posttext:
                self.sendLine(l)
            self.transport.loseConnection()

    def connectionLost(self, reason):
        self.deferred.callback(None)