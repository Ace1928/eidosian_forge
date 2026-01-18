from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer
class DummyClock:
    time = 0

    def set(self, when: int) -> None:
        self.time = when

    def __call__(self) -> int:
        return self.time