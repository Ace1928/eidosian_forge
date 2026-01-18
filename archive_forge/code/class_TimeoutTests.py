from __future__ import annotations
import unittest as pyunit
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.trial import reporter, unittest, util
from twisted.trial.test import detests
class TimeoutTests(TestTester):

    def getTest(self, name: str) -> detests.TimeoutTests:
        return detests.TimeoutTests(name)

    def _wasTimeout(self, error: Failure) -> None:
        self.assertEqual(error.check(defer.TimeoutError), defer.TimeoutError)

    def test_pass(self) -> None:
        result = self.runTest('test_pass')
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)

    def test_passDefault(self) -> None:
        result = self.runTest('test_passDefault')
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)

    def test_timeout(self) -> None:
        result = self.runTest('test_timeout')
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.errors), 1)
        assert isinstance(result.errors[0][1], Failure)
        self._wasTimeout(result.errors[0][1])

    def test_timeoutZero(self) -> None:
        result = self.runTest('test_timeoutZero')
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.errors), 1)
        assert isinstance(result.errors[0][1], Failure)
        self._wasTimeout(result.errors[0][1])

    def test_skip(self) -> None:
        result = self.runTest('test_skip')
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.skips), 1)

    def test_todo(self) -> None:
        result = self.runTest('test_expectedFailure')
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.expectedFailures), 1)
        assert isinstance(result.expectedFailures[0][1], Failure)
        self._wasTimeout(result.expectedFailures[0][1])

    def test_errorPropagation(self) -> None:
        result = self.runTest('test_errorPropagation')
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        assert detests.TimeoutTests.timedOut is not None
        self._wasTimeout(detests.TimeoutTests.timedOut)

    def test_classTimeout(self) -> None:
        loader = pyunit.TestLoader()
        suite = loader.loadTestsFromTestCase(detests.TestClassTimeoutAttribute)
        result = reporter.TestResult()
        suite.run(result)
        self.assertEqual(len(result.errors), 1)
        assert isinstance(result.errors[0][1], Failure)
        self._wasTimeout(result.errors[0][1])

    def test_callbackReturnsNonCallingDeferred(self) -> None:
        from twisted.internet import reactor
        call = reactor.callLater(2, reactor.crash)
        result = self.runTest('test_calledButNeverCallback')
        if call.active():
            call.cancel()
        self.assertFalse(result.wasSuccessful())
        assert isinstance(result.errors[0][1], Failure)
        self._wasTimeout(result.errors[0][1])