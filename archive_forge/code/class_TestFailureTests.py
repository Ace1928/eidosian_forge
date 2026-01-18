import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
class TestFailureTests(pyunit.TestCase):
    """
    Tests for the most basic functionality of L{SynchronousTestCase}, for
    failing tests.

    This class contains tests to demonstrate that L{SynchronousTestCase.fail}
    can be used to fail a test, and that that failure is reflected in the test
    result object.  This should be sufficient functionality so that further
    tests can be built on L{SynchronousTestCase} instead of
    L{unittest.TestCase}.  This depends on L{unittest.TestCase} working.
    """

    class FailingTest(unittest.SynchronousTestCase):

        def test_fails(self):
            self.fail('This test fails.')

    def setUp(self):
        """
        Load a suite of one test which can be used to exercise the failure
        handling behavior.
        """
        components = [__name__, self.__class__.__name__, self.FailingTest.__name__]
        self.loader = pyunit.TestLoader()
        self.suite = self.loader.loadTestsFromName('.'.join(components))
        self.test = list(self.suite)[0]

    def test_fail(self):
        """
        L{SynchronousTestCase.fail} raises
        L{SynchronousTestCase.failureException} with the given argument.
        """
        try:
            self.test.fail('failed')
        except self.test.failureException as result:
            self.assertEqual('failed', str(result))
        else:
            self.fail('SynchronousTestCase.fail method did not raise SynchronousTestCase.failureException')

    def test_failingExceptionFails(self):
        """
        When a test method raises L{SynchronousTestCase.failureException}, the test is
        marked as having failed on the L{TestResult}.
        """
        result = pyunit.TestResult()
        self.suite.run(result)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0][0], self.test)