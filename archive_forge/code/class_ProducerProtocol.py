from twisted.internet import interfaces
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import TCPCreator
from twisted.internet.test.test_tls import (
from twisted.trial import unittest
from zope.interface import implementer
@implementer(interfaces.IHandshakeListener)
class ProducerProtocol(ConnectableProtocol):
    """
    Register a producer, unregister it, and verify the producer hooks up to
    innards of C{TLSMemoryBIOProtocol}.
    """

    def __init__(self, producer, result):
        self.producer = producer
        self.result = result

    def handshakeCompleted(self):
        if not isinstance(self.transport.protocol, tls.BufferingTLSTransport):
            raise RuntimeError('TLSMemoryBIOProtocol not hooked up.')
        self.transport.registerProducer(self.producer, True)
        self.result.append(self.transport.protocol._producer._producer)
        self.transport.unregisterProducer()
        self.result.append(self.transport.protocol._producer)
        self.transport.loseConnection()