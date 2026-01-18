from twisted.protocols import pcp
from twisted.trial import unittest
class ProducerInterfaceTest:
    """Test ProducerConsumerProxy as a Producer.

    Normally we have ProducingServer -> ConsumingTransport.

    If I am to go between (Server -> Shaper -> Transport), I have to
    play the role of Producer convincingly for the ConsumingTransport.
    """

    def setUp(self):
        self.consumer = DummyConsumer()
        self.producer = self.proxyClass(self.consumer)

    def testRegistersProducer(self):
        self.assertEqual(self.consumer.producer[0], self.producer)

    def testPause(self):
        self.producer.pauseProducing()
        self.producer.write('yakkity yak')
        self.assertFalse(self.consumer.getvalue(), 'Paused producer should not have sent data.')

    def testResume(self):
        self.producer.pauseProducing()
        self.producer.resumeProducing()
        self.producer.write('yakkity yak')
        self.assertEqual(self.consumer.getvalue(), 'yakkity yak')

    def testResumeNoEmptyWrite(self):
        self.producer.pauseProducing()
        self.producer.resumeProducing()
        self.assertEqual(len(self.consumer._writes), 0, 'Resume triggered an empty write.')

    def testResumeBuffer(self):
        self.producer.pauseProducing()
        self.producer.write('buffer this')
        self.producer.resumeProducing()
        self.assertEqual(self.consumer.getvalue(), 'buffer this')

    def testStop(self):
        self.producer.stopProducing()
        self.producer.write('yakkity yak')
        self.assertFalse(self.consumer.getvalue(), 'Stopped producer should not have sent data.')