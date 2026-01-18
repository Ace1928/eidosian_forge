from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def _producertest(self, producerClass):
    toProduce = [b'%d' % (i,) for i in range(0, 10)]

    class ProducingProtocol(Protocol):

        def connectionMade(self):
            self.producer = producerClass(list(toProduce))
            self.producer.start(self.transport)

    class ReceivingProtocol(Protocol):
        bytes = b''

        def dataReceived(self, data):
            self.bytes += data
            if self.bytes == b''.join(toProduce):
                self.received.callback((client, server))
    server = ProducingProtocol()
    client = ReceivingProtocol()
    client.received = Deferred()
    loopback.loopbackAsync(server, client)
    return client.received