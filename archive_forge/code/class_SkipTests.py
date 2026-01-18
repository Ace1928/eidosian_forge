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
class SkipTests(unittest.SynchronousTestCase):
    """
    Tests for L{reporter.Reporter}'s handling of skips.
    """

    def setUp(self):
        self.stream = StringIO()
        self.result = reporter.Reporter(self.stream)
        self.test = sample.FooTest('test_foo')

    def _getSkips(self, result):
        """
        Get the number of skips that happened to a reporter.
        """
        return len(result.skips)

    def test_accumulation(self):
        self.result.addSkip(self.test, 'some reason')
        self.assertEqual(self._getSkips(self.result), 1)

    def test_success(self):
        self.result.addSkip(self.test, 'some reason')
        self.assertEqual(True, self.result.wasSuccessful())

    def test_summary(self):
        """
        The summary of a successful run with skips indicates that the test
        suite passed and includes the number of skips.
        """
        self.result.addSkip(self.test, 'some reason')
        self.result.done()
        output = self.stream.getvalue().splitlines()[-1]
        prefix = 'PASSED '
        self.assertTrue(output.startswith(prefix))
        self.assertEqual(output[len(prefix):].strip(), '(skips=1)')

    def test_basicErrors(self):
        """
        The output at the end of a test run with skips includes the reasons
        for skipping those tests.
        """
        self.result.addSkip(self.test, 'some reason')
        self.result.done()
        output = self.stream.getvalue().splitlines()[3]
        self.assertEqual(output.strip(), 'some reason')

    def test_booleanSkip(self):
        """
        Tests can be skipped without specifying a reason by setting the 'skip'
        attribute to True. When this happens, the test output includes 'True'
        as the reason.
        """
        self.result.addSkip(self.test, True)
        self.result.done()
        output = self.stream.getvalue().splitlines()[3]
        self.assertEqual(output, 'True')

    def test_exceptionSkip(self):
        """
        Skips can be raised as errors. When this happens, the error is
        included in the summary at the end of the test suite.
        """
        try:
            1 / 0
        except Exception as e:
            error = e
        self.result.addSkip(self.test, error)
        self.result.done()
        output = '\n'.join(self.stream.getvalue().splitlines()[3:5]).strip()
        self.assertEqual(output, str(error))