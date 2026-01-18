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
class DirtyReactorTests(unittest.SynchronousTestCase):
    """
    The trial script has an option to treat L{DirtyReactorAggregateError}s as
    warnings, as a migration tool for test authors. It causes a wrapper to be
    placed around reporters that replaces L{DirtyReactorAggregatErrors} with
    warnings.
    """

    def setUp(self):
        self.dirtyError = Failure(util.DirtyReactorAggregateError(['foo'], ['bar']))
        self.output = StringIO()
        self.test = DirtyReactorTests('test_errorByDefault')

    def test_errorByDefault(self):
        """
        L{DirtyReactorAggregateError}s are reported as errors with the default
        Reporter.
        """
        result = reporter.Reporter(stream=self.output)
        result.addError(self.test, self.dirtyError)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0][1], self.dirtyError)

    def test_warningsEnabled(self):
        """
        L{DirtyReactorAggregateError}s are reported as warnings when using
        the L{UncleanWarningsReporterWrapper}.
        """
        result = UncleanWarningsReporterWrapper(reporter.Reporter(stream=self.output))
        self.assertWarns(UserWarning, self.dirtyError.getErrorMessage(), reporter.__file__, result.addError, self.test, self.dirtyError)

    def test_warningsMaskErrors(self):
        """
        L{DirtyReactorAggregateError}s are I{not} reported as errors if the
        L{UncleanWarningsReporterWrapper} is used.
        """
        result = UncleanWarningsReporterWrapper(reporter.Reporter(stream=self.output))
        self.assertWarns(UserWarning, self.dirtyError.getErrorMessage(), reporter.__file__, result.addError, self.test, self.dirtyError)
        self.assertEqual(result._originalReporter.errors, [])

    def test_dealsWithThreeTuples(self):
        """
        Some annoying stuff can pass three-tuples to addError instead of
        Failures (like PyUnit). The wrapper, of course, handles this case,
        since it is a part of L{twisted.trial.itrial.IReporter}! But it does
        not convert L{DirtyReactorAggregateError} to warnings in this case,
        because nobody should be passing those in the form of three-tuples.
        """
        result = UncleanWarningsReporterWrapper(reporter.Reporter(stream=self.output))
        result.addError(self.test, (self.dirtyError.type, self.dirtyError.value, None))
        self.assertEqual(len(result._originalReporter.errors), 1)
        self.assertEqual(result._originalReporter.errors[0][1].type, self.dirtyError.type)
        self.assertEqual(result._originalReporter.errors[0][1].value, self.dirtyError.value)