from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
class GreeteeProtocol(Protocol):
    bytes = b''

    def dataReceived(self, bytes):
        self.bytes += bytes
        if self.bytes == b'bytes':
            self.received.callback(None)