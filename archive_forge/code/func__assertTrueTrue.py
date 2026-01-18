import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def _assertTrueTrue(self, method):
    """
        Perform the positive case test for C{assertTrue} and C{failUnless}.

        @param method: The test method to test.
        """
    for true in [1, True, 'cat', [1, 2], (3, 4)]:
        result = method(true, f'failed on {true!r}')
        if result != true:
            self.fail(f'Did not return argument {true!r}')