import unittest as pyunit
from twisted.internet import defer
from twisted.python import failure
from twisted.trial import unittest
class TC(unittest.TestCase):
    failureException = ExampleFailure

    def test_assertFailure(self):
        d = defer.maybeDeferred(lambda: 1 / 0)
        self.assertFailure(d, OverflowError)
        self.assertFailure(d, ZeroDivisionError)
        return d