from twisted.protocols import pcp
from twisted.trial import unittest
class BufferedPullTests(unittest.TestCase):

    class proxyClass(pcp.ProducerConsumerProxy):
        iAmStreaming = False

        def _writeSomeData(self, data):
            pcp.ProducerConsumerProxy._writeSomeData(self, data[:100])
            return min(len(data), 100)

    def setUp(self):
        self.underlying = DummyConsumer()
        self.proxy = self.proxyClass(self.underlying)
        self.proxy.bufferSize = 100
        self.parentProducer = DummyProducer(self.proxy)
        self.proxy.registerProducer(self.parentProducer, False)

    def testResumePull(self):
        self.parentProducer.resumed = False
        self.proxy.resumeProducing()
        self.assertTrue(self.parentProducer.resumed)

    def testLateWriteBuffering(self):
        self.proxy.resumeProducing()
        self.proxy.write('datum' * 21)
        self.assertEqual(self.underlying.getvalue(), 'datum' * 20)
        self.assertEqual(self.proxy._buffer, ['datum'])