from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
class ProducingProtocol(Protocol):

    def connectionMade(self):
        self.producer = producerClass(list(toProduce))
        self.producer.start(self.transport)