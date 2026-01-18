import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def _testUnequalPair(self, first, second):
    """
        Assert that when called with unequal arguments, C{assertEqual} raises a
        failure exception with the same message as the standard library
        C{assertEqual} would have raised.
        """
    raised = False
    try:
        self.assertEqual(first, second)
    except self.failureException as ourFailure:
        case = pyunit.TestCase('setUp')
        try:
            case.assertEqual(first, second)
        except case.failureException as theirFailure:
            raised = True
            got = str(ourFailure)
            expected = str(theirFailure)
            if expected != got:
                self.fail(f'Expected: {expected!r}; Got: {got!r}')
    if not raised:
        self.fail(f"Call to assertEqual({first!r}, {second!r}) didn't fail")