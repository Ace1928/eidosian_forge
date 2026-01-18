from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer
class BucketTests(TestBucketBase):

    def testBucketSize(self) -> None:
        """
        Testing the size of the bucket.
        """
        b = SomeBucket()
        fit = b.add(1000)
        self.assertEqual(100, fit)

    def testBucketDrain(self) -> None:
        """
        Testing the bucket's drain rate.
        """
        b = SomeBucket()
        fit = b.add(1000)
        self.clock.set(10)
        fit = b.add(1000)
        self.assertEqual(20, fit)

    def test_bucketEmpty(self) -> None:
        """
        L{htb.Bucket.drip} returns C{True} if the bucket is empty after that drip.
        """
        b = SomeBucket()
        b.add(20)
        self.clock.set(9)
        empty = b.drip()
        self.assertFalse(empty)
        self.clock.set(10)
        empty = b.drip()
        self.assertTrue(empty)