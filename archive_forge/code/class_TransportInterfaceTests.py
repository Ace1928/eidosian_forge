from twisted.protocols import pcp
from twisted.trial import unittest
class TransportInterfaceTests(unittest.TestCase):
    proxyClass = pcp.BasicProducerConsumerProxy

    def setUp(self):
        self.underlying = DummyConsumer()
        self.transport = self.proxyClass(self.underlying)

    def testWrite(self):
        self.transport.write('some bytes')