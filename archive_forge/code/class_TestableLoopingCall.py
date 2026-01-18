from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
class TestableLoopingCall(task.LoopingCall):

    def __init__(self, clock, *a, **kw):
        super().__init__(*a, **kw)
        self.clock = clock