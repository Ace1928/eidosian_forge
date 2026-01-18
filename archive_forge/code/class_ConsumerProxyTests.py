from twisted.protocols import pcp
from twisted.trial import unittest
class ConsumerProxyTests(unittest.TestCase):
    """Consumer methods on me should be relayed to the Consumer I proxy."""
    proxyClass = pcp.BasicProducerConsumerProxy

    def setUp(self):
        self.underlying = DummyConsumer()
        self.consumer = self.proxyClass(self.underlying)

    def testWrite(self):
        self.consumer.write('some bytes')
        self.assertEqual(self.underlying.getvalue(), 'some bytes')

    def testFinish(self):
        self.consumer.finish()
        self.assertTrue(self.underlying.finished)

    def testUnregister(self):
        self.consumer.unregisterProducer()
        self.assertTrue(self.underlying.unregistered)