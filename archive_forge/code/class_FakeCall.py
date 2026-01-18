from twisted.internet import defer, reactor, task
from twisted.trial import unittest
class FakeCall:

    def __init__(self, func):
        self.func = func

    def __repr__(self) -> str:
        return f'<FakeCall {self.func!r}>'