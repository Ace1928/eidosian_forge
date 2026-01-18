from __future__ import annotations
import unittest as pyunit
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.trial import reporter, unittest, util
from twisted.trial.test import detests
class DeferredTests(TestTester):

    def getTest(self, name: str) -> detests.DeferredTests:
        return detests.DeferredTests(name)

    def test_pass(self) -> None:
        result = self.runTest('test_pass')
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)

    def test_passGenerated(self) -> None:
        result = self.runTest('test_passGenerated')
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertTrue(detests.DeferredTests.touched)
    test_passGenerated.supress = [util.suppress(message='twisted.internet.defer.deferredGenerator is deprecated')]

    def test_passInlineCallbacks(self) -> None:
        """
        The body of a L{defer.inlineCallbacks} decorated test gets run.
        """
        result = self.runTest('test_passInlineCallbacks')
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertTrue(detests.DeferredTests.touched)

    def test_fail(self) -> None:
        result = self.runTest('test_fail')
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.failures), 1)

    def test_failureInCallback(self) -> None:
        result = self.runTest('test_failureInCallback')
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.failures), 1)

    def test_errorInCallback(self) -> None:
        result = self.runTest('test_errorInCallback')
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.errors), 1)

    def test_skip(self) -> None:
        result = self.runTest('test_skip')
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.skips), 1)
        self.assertFalse(detests.DeferredTests.touched)

    def test_todo(self) -> None:
        result = self.runTest('test_expectedFailure')
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(len(result.expectedFailures), 1)

    def test_thread(self) -> None:
        result = self.runTest('test_thread')
        self.assertEqual(result.testsRun, 1)
        self.assertTrue(result.wasSuccessful(), result.errors)