from twisted.protocols import pcp
from twisted.trial import unittest
class BufferedConsumerTests(unittest.TestCase):
    """As a consumer, ask the producer to pause after too much data."""
    proxyClass = pcp.ProducerConsumerProxy

    def setUp(self):
        self.underlying = DummyConsumer()
        self.proxy = self.proxyClass(self.underlying)
        self.proxy.bufferSize = 100
        self.parentProducer = DummyProducer(self.proxy)
        self.proxy.registerProducer(self.parentProducer, True)

    def testRegisterPull(self):
        self.proxy.registerProducer(self.parentProducer, False)
        self.assertTrue(self.parentProducer.resumed)

    def testPauseIntercept(self):
        self.proxy.pauseProducing()
        self.assertFalse(self.parentProducer.paused)

    def testResumeIntercept(self):
        self.proxy.pauseProducing()
        self.proxy.resumeProducing()
        self.assertFalse(self.parentProducer.resumed)

    def testTriggerPause(self):
        """Make sure I say "when." """
        self.proxy.pauseProducing()
        self.assertFalse(self.parentProducer.paused, "don't pause yet")
        self.proxy.write('x' * 51)
        self.assertFalse(self.parentProducer.paused, "don't pause yet")
        self.proxy.write('x' * 51)
        self.assertTrue(self.parentProducer.paused)

    def testTriggerResume(self):
        """Make sure I resumeProducing when my buffer empties."""
        self.proxy.pauseProducing()
        self.proxy.write('x' * 102)
        self.assertTrue(self.parentProducer.paused, 'should be paused')
        self.proxy.resumeProducing()
        self.assertFalse(self.parentProducer.paused, 'Producer should have resumed.')
        self.assertFalse(self.proxy.producerPaused)