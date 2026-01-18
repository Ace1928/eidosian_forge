import sys
from twisted.internet import protocol, stdio
from twisted.python import reflect
class HostPeerChild(protocol.Protocol):

    def connectionMade(self):
        self.transport.write(b'\n'.join([str(self.transport.getHost()).encode('ascii'), str(self.transport.getPeer()).encode('ascii')]))
        self.transport.loseConnection()

    def connectionLost(self, reason):
        reactor.stop()