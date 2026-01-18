import sys
from twisted.internet import protocol, stdio
from twisted.python import reflect
class WriteChild(protocol.Protocol):

    def connectionMade(self):
        self.transport.write(b'o')
        self.transport.write(b'k')
        self.transport.write(b'!')
        self.transport.loseConnection()

    def connectionLost(self, reason):
        reactor.stop()