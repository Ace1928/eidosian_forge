from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
@implementer(IPullProducer)
class PullProducer:

    def __init__(self, toProduce):
        self.toProduce = toProduce

    def start(self, consumer):
        self.consumer = consumer
        self.consumer.registerProducer(self, False)

    def resumeProducing(self):
        self.consumer.write(self.toProduce.pop(0))
        if not self.toProduce:
            self.consumer.unregisterProducer()