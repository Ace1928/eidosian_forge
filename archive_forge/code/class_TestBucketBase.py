from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer
class TestBucketBase(unittest.TestCase):

    def setUp(self) -> None:
        self._realTimeFunc = htb.time
        self.clock = DummyClock()
        htb.time = self.clock

    def tearDown(self) -> None:
        htb.time = self._realTimeFunc