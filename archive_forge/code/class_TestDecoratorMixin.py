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
class TestDecoratorMixin:
    """
    Tests for our test decoration features.
    """

    def assertTestsEqual(self, observed, expected):
        """
        Assert that the given decorated tests are equal.
        """
        self.assertEqual(observed.__class__, expected.__class__, 'Different class')
        observedOriginal = getattr(observed, '_originalTest', None)
        expectedOriginal = getattr(expected, '_originalTest', None)
        self.assertIdentical(observedOriginal, expectedOriginal)
        if observedOriginal is expectedOriginal is None:
            self.assertIdentical(observed, expected)

    def assertSuitesEqual(self, observed, expected):
        """
        Assert that the given test suites with decorated tests are equal.
        """
        self.assertEqual(observed.__class__, expected.__class__, 'Different class')
        self.assertEqual(len(observed._tests), len(expected._tests), 'Different number of tests.')
        for observedTest, expectedTest in zip(observed._tests, expected._tests):
            if getattr(observedTest, '_tests', None) is not None:
                self.assertSuitesEqual(observedTest, expectedTest)
            else:
                self.assertTestsEqual(observedTest, expectedTest)

    def test_usesAdaptedReporterWithRun(self):
        """
        For decorated tests, C{run} uses a result adapter that preserves the
        test decoration for calls to C{addError}, C{startTest} and the like.

        See L{reporter._AdaptedReporter}.
        """
        test = self.TestCase()
        decoratedTest = unittest.TestDecorator(test)
        from twisted.trial.test.test_reporter import LoggingReporter
        result = LoggingReporter()
        decoratedTest.run(result)
        self.assertTestsEqual(result.test, decoratedTest)

    def test_usesAdaptedReporterWithCall(self):
        """
        For decorated tests, C{__call__} uses a result adapter that preserves
        the test decoration for calls to C{addError}, C{startTest} and the
        like.

        See L{reporter._AdaptedReporter}.
        """
        test = self.TestCase()
        decoratedTest = unittest.TestDecorator(test)
        from twisted.trial.test.test_reporter import LoggingReporter
        result = LoggingReporter()
        decoratedTest(result)
        self.assertTestsEqual(result.test, decoratedTest)

    def test_decorateSingleTest(self):
        """
        Calling L{decorate} on a single test case returns the test case
        decorated with the provided decorator.
        """
        test = self.TestCase()
        decoratedTest = unittest.decorate(test, unittest.TestDecorator)
        self.assertTestsEqual(unittest.TestDecorator(test), decoratedTest)

    def test_decorateTestSuite(self):
        """
        Calling L{decorate} on a test suite will return a test suite with
        each test decorated with the provided decorator.
        """
        test = self.TestCase()
        suite = unittest.TestSuite([test])
        decoratedTest = unittest.decorate(suite, unittest.TestDecorator)
        self.assertSuitesEqual(decoratedTest, unittest.TestSuite([unittest.TestDecorator(test)]))

    def test_decorateInPlaceMutatesOriginal(self):
        """
        Calling L{decorate} on a test suite will mutate the original suite.
        """
        test = self.TestCase()
        suite = unittest.TestSuite([test])
        decoratedTest = unittest.decorate(suite, unittest.TestDecorator)
        self.assertSuitesEqual(decoratedTest, unittest.TestSuite([unittest.TestDecorator(test)]))
        self.assertSuitesEqual(suite, unittest.TestSuite([unittest.TestDecorator(test)]))

    def test_decorateTestSuiteReferences(self):
        """
        When decorating a test suite in-place, the number of references to the
        test objects in that test suite should stay the same.

        Previously, L{unittest.decorate} recreated a test suite, so the
        original suite kept references to the test objects. This test is here
        to ensure the problem doesn't reappear again.
        """
        getrefcount = getattr(sys, 'getrefcount', None)
        if getrefcount is None:
            raise unittest.SkipTest('getrefcount not supported on this platform')
        test = self.TestCase()
        suite = unittest.TestSuite([test])
        count1 = getrefcount(test)
        unittest.decorate(suite, unittest.TestDecorator)
        count2 = getrefcount(test)
        self.assertEqual(count1, count2)

    def test_decorateNestedTestSuite(self):
        """
        Calling L{decorate} on a test suite with nested suites will return a
        test suite that maintains the same structure, but with all tests
        decorated.
        """
        test = self.TestCase()
        suite = unittest.TestSuite([unittest.TestSuite([test])])
        decoratedTest = unittest.decorate(suite, unittest.TestDecorator)
        expected = unittest.TestSuite([unittest.TestSuite([unittest.TestDecorator(test)])])
        self.assertSuitesEqual(decoratedTest, expected)

    def test_decorateDecoratedSuite(self):
        """
        Calling L{decorate} on a test suite with already-decorated tests
        decorates all of the tests in the suite again.
        """
        test = self.TestCase()
        decoratedTest = unittest.decorate(test, unittest.TestDecorator)
        redecoratedTest = unittest.decorate(decoratedTest, unittest.TestDecorator)
        self.assertTestsEqual(redecoratedTest, unittest.TestDecorator(decoratedTest))

    def test_decoratePreservesSuite(self):
        """
        Tests can be in non-standard suites. L{decorate} preserves the
        non-standard suites when it decorates the tests.
        """
        test = self.TestCase()
        suite = runner.DestructiveTestSuite([test])
        decorated = unittest.decorate(suite, unittest.TestDecorator)
        self.assertSuitesEqual(decorated, runner.DestructiveTestSuite([unittest.TestDecorator(test)]))