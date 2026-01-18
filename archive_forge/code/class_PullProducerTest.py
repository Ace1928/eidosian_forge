from twisted.protocols import pcp
from twisted.trial import unittest
class PullProducerTest:

    def setUp(self):
        self.underlying = DummyConsumer()
        self.proxy = self.proxyClass(self.underlying)
        self.parentProducer = DummyProducer(self.proxy)
        self.proxy.registerProducer(self.parentProducer, True)

    def testHoldWrites(self):
        self.proxy.write('hello')
        self.assertFalse(self.underlying.getvalue(), 'Pulling Consumer got data before it pulled.')

    def testPull(self):
        self.proxy.write('hello')
        self.proxy.resumeProducing()
        self.assertEqual(self.underlying.getvalue(), 'hello')

    def testMergeWrites(self):
        self.proxy.write('hello ')
        self.proxy.write('sunshine')
        self.proxy.resumeProducing()
        nwrites = len(self.underlying._writes)
        self.assertEqual(nwrites, 1, 'Pull resulted in %d writes instead of 1.' % (nwrites,))
        self.assertEqual(self.underlying.getvalue(), 'hello sunshine')

    def testLateWrite(self):
        self.proxy.resumeProducing()
        self.proxy.write('data')
        self.assertEqual(self.underlying.getvalue(), 'data')