import errno
import os
import re
import sys
from inspect import getmro
from io import BytesIO, StringIO
from typing import Type
from unittest import (
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.python import log
from twisted.python.failure import Failure
from twisted.trial import itrial, reporter, runner, unittest, util
from twisted.trial.reporter import UncleanWarningsReporterWrapper, _ExitWrapper
from twisted.trial.test import erroneous, sample
from twisted.trial.unittest import SkipTest, Todo, makeTodo
from .._dist.test.matchers import isFailure, matches_result, similarFrame
from .matchers import after
class ReporterInterfaceTests(unittest.SynchronousTestCase):
    """
    Tests for the bare interface of a trial reporter.

    Subclass this test case and provide a different 'resultFactory' to test
    that a particular reporter implementation will work with the rest of
    Trial.

    @cvar resultFactory: A callable that returns a reporter to be tested. The
        callable must take the same parameters as L{reporter.Reporter}.
    """
    resultFactory: Type[itrial.IReporter] = reporter.Reporter

    def setUp(self):
        self.test = sample.FooTest('test_foo')
        self.stream = StringIO()
        self.publisher = log.LogPublisher()
        self.result = self.resultFactory(self.stream, publisher=self.publisher)

    def test_shouldStopInitiallyFalse(self):
        """
        shouldStop is False to begin with.
        """
        self.assertEqual(False, self.result.shouldStop)

    def test_shouldStopTrueAfterStop(self):
        """
        shouldStop becomes True soon as someone calls stop().
        """
        self.result.stop()
        self.assertEqual(True, self.result.shouldStop)

    def test_wasSuccessfulInitiallyTrue(self):
        """
        wasSuccessful() is True when there have been no results reported.
        """
        self.assertEqual(True, self.result.wasSuccessful())

    def test_wasSuccessfulTrueAfterSuccesses(self):
        """
        wasSuccessful() is True when there have been only successes, False
        otherwise.
        """
        self.result.addSuccess(self.test)
        self.assertEqual(True, self.result.wasSuccessful())

    def test_wasSuccessfulFalseAfterErrors(self):
        """
        wasSuccessful() becomes False after errors have been reported.
        """
        try:
            1 / 0
        except ZeroDivisionError:
            self.result.addError(self.test, sys.exc_info())
        self.assertEqual(False, self.result.wasSuccessful())

    def test_wasSuccessfulFalseAfterFailures(self):
        """
        wasSuccessful() becomes False after failures have been reported.
        """
        try:
            self.fail('foo')
        except self.failureException:
            self.result.addFailure(self.test, sys.exc_info())
        self.assertEqual(False, self.result.wasSuccessful())