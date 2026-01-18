from __future__ import annotations
from twisted.internet import defer, reactor, threads
from twisted.python.failure import Failure
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
class DeferredSetUpOK(unittest.TestCase):

    def setUp(self):
        d = defer.succeed('value')
        d.addCallback(self._cb_setUpCalled)
        return d

    def _cb_setUpCalled(self, ignored):
        self._setUpCalled = True

    def test_ok(self):
        self.assertTrue(self._setUpCalled)