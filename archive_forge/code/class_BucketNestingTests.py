from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer
class BucketNestingTests(TestBucketBase):

    def setUp(self) -> None:
        TestBucketBase.setUp(self)
        self.parent = SomeBucket()
        self.child1 = SomeBucket(self.parent)
        self.child2 = SomeBucket(self.parent)

    def testBucketParentSize(self) -> None:
        self.child1.add(90)
        fit = self.child2.add(90)
        self.assertEqual(10, fit)

    def testBucketParentRate(self) -> None:
        self.parent.rate = 1
        self.child1.add(100)
        self.clock.set(10)
        fit = self.child1.add(100)
        self.assertEqual(10, fit)