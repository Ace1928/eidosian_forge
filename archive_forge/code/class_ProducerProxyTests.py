from twisted.protocols import pcp
from twisted.trial import unittest
class ProducerProxyTests(unittest.TestCase):
    """Producer methods on me should be relayed to the Producer I proxy."""
    proxyClass = pcp.BasicProducerConsumerProxy

    def setUp(self):
        self.proxy = self.proxyClass(None)
        self.parentProducer = DummyProducer(self.proxy)
        self.proxy.registerProducer(self.parentProducer, True)

    def testStop(self):
        self.proxy.stopProducing()
        self.assertTrue(self.parentProducer.stopped)