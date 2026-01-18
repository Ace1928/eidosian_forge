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
class AdaptedReporterTests(unittest.SynchronousTestCase):
    """
    L{reporter._AdaptedReporter} is a reporter wrapper that wraps all of the
    tests it receives before passing them on to the original reporter.
    """

    def setUp(self):
        self.wrappedResult = self.getWrappedResult()

    def _testAdapter(self, test):
        return test.id()

    def assertWrapped(self, wrappedResult, test):
        self.assertEqual(wrappedResult._originalReporter.test, self._testAdapter(test))

    def getFailure(self, exceptionInstance):
        """
        Return a L{Failure} from raising the given exception.

        @param exceptionInstance: The exception to raise.
        @return: L{Failure}
        """
        try:
            raise exceptionInstance
        except BaseException:
            return Failure()

    def getWrappedResult(self):
        result = LoggingReporter()
        return reporter._AdaptedReporter(result, self._testAdapter)

    def test_addError(self):
        """
        C{addError} wraps its test with the provided adapter.
        """
        self.wrappedResult.addError(self, self.getFailure(RuntimeError()))
        self.assertWrapped(self.wrappedResult, self)

    def test_addFailure(self):
        """
        C{addFailure} wraps its test with the provided adapter.
        """
        self.wrappedResult.addFailure(self, self.getFailure(AssertionError()))
        self.assertWrapped(self.wrappedResult, self)

    def test_addSkip(self):
        """
        C{addSkip} wraps its test with the provided adapter.
        """
        self.wrappedResult.addSkip(self, self.getFailure(SkipTest('no reason')))
        self.assertWrapped(self.wrappedResult, self)

    def test_startTest(self):
        """
        C{startTest} wraps its test with the provided adapter.
        """
        self.wrappedResult.startTest(self)
        self.assertWrapped(self.wrappedResult, self)

    def test_stopTest(self):
        """
        C{stopTest} wraps its test with the provided adapter.
        """
        self.wrappedResult.stopTest(self)
        self.assertWrapped(self.wrappedResult, self)

    def test_addExpectedFailure(self):
        """
        C{addExpectedFailure} wraps its test with the provided adapter.
        """
        self.wrappedResult.addExpectedFailure(self, self.getFailure(RuntimeError()), Todo('no reason'))
        self.assertWrapped(self.wrappedResult, self)

    def test_expectedFailureWithoutTodo(self):
        """
        C{addExpectedFailure} works without a C{Todo}.
        """
        self.wrappedResult.addExpectedFailure(self, self.getFailure(RuntimeError()))
        self.assertWrapped(self.wrappedResult, self)

    def test_addUnexpectedSuccess(self):
        """
        C{addUnexpectedSuccess} wraps its test with the provided adapter.
        """
        self.wrappedResult.addUnexpectedSuccess(self, Todo('no reason'))
        self.assertWrapped(self.wrappedResult, self)

    def test_unexpectedSuccessWithoutTodo(self):
        """
        C{addUnexpectedSuccess} works without a C{Todo}.
        """
        self.wrappedResult.addUnexpectedSuccess(self)
        self.assertWrapped(self.wrappedResult, self)