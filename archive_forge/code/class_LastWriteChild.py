import sys
from twisted.internet.protocol import Protocol
from twisted.internet.stdio import StandardIO
from twisted.python.reflect import namedAny
class LastWriteChild(Protocol):

    def __init__(self, reactor, magicString):
        self.reactor = reactor
        self.magicString = magicString

    def connectionMade(self):
        self.transport.write(self.magicString)
        self.transport.loseConnection()

    def connectionLost(self, reason):
        self.reactor.stop()