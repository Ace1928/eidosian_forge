import sys
from twisted.internet import protocol, stdio
from twisted.python import reflect
class WriteSequenceChild(protocol.Protocol):

    def connectionMade(self):
        self.transport.writeSequence([b'o', b'k', b'!'])
        self.transport.loseConnection()

    def connectionLost(self, reason):
        reactor.stop()