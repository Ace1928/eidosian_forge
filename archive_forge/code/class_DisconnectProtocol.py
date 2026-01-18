from twisted.internet.protocol import Protocol
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.runtime import platform
class DisconnectProtocol(Protocol):

    def connectionLost(self, reason):
        reactor.stop()