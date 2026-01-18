import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
class AssertTrueTests(unittest.SynchronousTestCase):
    """
    Tests for L{SynchronousTestCase}'s C{assertTrue} and C{failUnless} assertion
    methods.

    This is pretty paranoid.  Still, a certain paranoia is healthy if you
    are testing a unit testing framework.

    @note: As of 11.2, C{assertTrue} is preferred over C{failUnless}.
    """

    def _assertTrueFalse(self, method):
        """
        Perform the negative case test for C{assertTrue} and C{failUnless}.

        @param method: The test method to test.
        """
        for notTrue in [0, 0.0, False, None, (), []]:
            try:
                method(notTrue, f'failed on {notTrue!r}')
            except self.failureException as e:
                self.assertIn(f'failed on {notTrue!r}', str(e), f'Raised incorrect exception on {notTrue!r}: {e!r}')
            else:
                self.fail("Call to %s(%r) didn't fail" % (method.__name__, notTrue))

    def _assertTrueTrue(self, method):
        """
        Perform the positive case test for C{assertTrue} and C{failUnless}.

        @param method: The test method to test.
        """
        for true in [1, True, 'cat', [1, 2], (3, 4)]:
            result = method(true, f'failed on {true!r}')
            if result != true:
                self.fail(f'Did not return argument {true!r}')

    def test_assertTrueFalse(self):
        """
        L{SynchronousTestCase.assertTrue} raises
        L{SynchronousTestCase.failureException} if its argument is not
        considered true.
        """
        self._assertTrueFalse(self.assertTrue)

    def test_failUnlessFalse(self):
        """
        L{SynchronousTestCase.failUnless} raises
        L{SynchronousTestCase.failureException} if its argument is not
        considered true.
        """
        self._assertTrueFalse(self.failUnless)

    def test_assertTrueTrue(self):
        """
        L{SynchronousTestCase.assertTrue} returns its argument if its argument
        is considered true.
        """
        self._assertTrueTrue(self.assertTrue)

    def test_failUnlessTrue(self):
        """
        L{SynchronousTestCase.failUnless} returns its argument if its argument
        is considered true.
        """
        self._assertTrueTrue(self.failUnless)