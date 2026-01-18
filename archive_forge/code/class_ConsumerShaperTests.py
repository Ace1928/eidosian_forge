from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer
class ConsumerShaperTests(TestBucketBase):

    def setUp(self) -> None:
        TestBucketBase.setUp(self)
        self.underlying = DummyConsumer()
        self.bucket = SomeBucket()
        self.shaped = htb.ShapedConsumer(self.underlying, self.bucket)

    def testRate(self) -> None:
        delta_t = 10
        self.bucket.add(100)
        self.shaped.write('x' * 100)
        self.clock.set(delta_t)
        self.shaped.resumeProducing()
        self.assertEqual(len(self.underlying.getvalue()), delta_t * self.bucket.rate)

    def testBucketRefs(self) -> None:
        self.assertEqual(self.bucket._refcount, 1)
        self.shaped.stopProducing()
        self.assertEqual(self.bucket._refcount, 0)