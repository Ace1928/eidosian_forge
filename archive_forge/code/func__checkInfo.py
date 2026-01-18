import unittest as pyunit
from twisted.internet import defer
from twisted.python import failure
from twisted.trial import unittest
def _checkInfo(self, assertionFailure, f):
    assert assertionFailure.check(self.failureException)
    output = assertionFailure.getErrorMessage()
    self.assertIn(f.getErrorMessage(), output)
    self.assertIn(f.getBriefTraceback(), output)