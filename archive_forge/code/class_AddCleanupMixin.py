import gc
import sys
import unittest as pyunit
import weakref
from io import StringIO
from twisted.internet import defer, reactor
from twisted.python.compat import _PYPY
from twisted.python.reflect import namedAny
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import (
from twisted.trial.test import erroneous
from twisted.trial.test.test_suppression import SuppressionMixin
class AddCleanupMixin:
    """
    Test the addCleanup method of TestCase.
    """

    def setUp(self):
        """
        Setup our test case
        """
        super().setUp()
        self.result = reporter.TestResult()
        self.test = self.AddCleanup()

    def test_addCleanupCalledIfSetUpFails(self):
        """
        Callables added with C{addCleanup} are run even if setUp fails.
        """
        self.test.setUp = self.test.brokenSetUp
        self.test.addCleanup(self.test.append, 'foo')
        self.test.run(self.result)
        self.assertEqual(['setUp', 'foo'], self.test.log)

    def test_addCleanupCalledIfSetUpSkips(self):
        """
        Callables added with C{addCleanup} are run even if setUp raises
        L{SkipTest}. This allows test authors to reliably provide clean up
        code using C{addCleanup}.
        """
        self.test.setUp = self.test.skippingSetUp
        self.test.addCleanup(self.test.append, 'foo')
        self.test.run(self.result)
        self.assertEqual(['setUp', 'foo'], self.test.log)

    def test_addCleanupCalledInReverseOrder(self):
        """
        Callables added with C{addCleanup} should be called before C{tearDown}
        in reverse order of addition.
        """
        self.test.addCleanup(self.test.append, 'foo')
        self.test.addCleanup(self.test.append, 'bar')
        self.test.run(self.result)
        self.assertEqual(['setUp', 'runTest', 'bar', 'foo', 'tearDown'], self.test.log)

    def test_errorInCleanupIsCaptured(self):
        """
        Errors raised in cleanup functions should be treated like errors in
        C{tearDown}. They should be added as errors and fail the test. Skips,
        todos and failures are all treated as errors.
        """
        self.test.addCleanup(self.test.fail, 'foo')
        self.test.run(self.result)
        self.assertFalse(self.result.wasSuccessful())
        self.assertEqual(1, len(self.result.errors))
        [(test, error)] = self.result.errors
        self.assertEqual(test, self.test)
        self.assertEqual(error.getErrorMessage(), 'foo')

    def test_cleanupsContinueRunningAfterError(self):
        """
        If a cleanup raises an error then that does not stop the other
        cleanups from being run.
        """
        self.test.addCleanup(self.test.append, 'foo')
        self.test.addCleanup(self.test.fail, 'bar')
        self.test.run(self.result)
        self.assertEqual(['setUp', 'runTest', 'foo', 'tearDown'], self.test.log)
        self.assertEqual(1, len(self.result.errors))
        [(test, error)] = self.result.errors
        self.assertEqual(test, self.test)
        self.assertEqual(error.getErrorMessage(), 'bar')

    def test_multipleErrorsReported(self):
        """
        If more than one cleanup fails, then the test should fail with more
        than one error.
        """
        self.test.addCleanup(self.test.fail, 'foo')
        self.test.addCleanup(self.test.fail, 'bar')
        self.test.run(self.result)
        self.assertEqual(['setUp', 'runTest', 'tearDown'], self.test.log)
        self.assertEqual(2, len(self.result.errors))
        [(test1, error1), (test2, error2)] = self.result.errors
        self.assertEqual(test1, self.test)
        self.assertEqual(test2, self.test)
        self.assertEqual(error1.getErrorMessage(), 'bar')
        self.assertEqual(error2.getErrorMessage(), 'foo')

    def test_cleanupRunsOnce(self):
        """
        A function registered as a cleanup is run once.
        """
        cleanups = []
        self.test.addCleanup(lambda: cleanups.append(stage))
        stage = 'first'
        self.test.run(self.result)
        stage = 'second'
        self.test.run(self.result)
        self.assertEqual(cleanups, ['first'])