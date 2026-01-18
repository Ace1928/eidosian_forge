from twisted.internet.error import ConnectionDone
from twisted.internet.protocol import Protocol
from twisted.python.failure import Failure
from twisted.trial import unittest
class SerialProtocol(Protocol):

    def connectionMade(self):
        events.append('connectionMade')

    def connectionLost(self, reason):
        events.append(('connectionLost', reason))