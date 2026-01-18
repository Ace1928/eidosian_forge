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
class UncleanWarningWrapperErrorReportingTests(ErrorReportingTests):
    """
    Tests that the L{UncleanWarningsReporterWrapper} can sufficiently proxy
    IReporter failure and error reporting methods to a L{reporter.Reporter}.
    """

    def setUp(self) -> None:
        self.loader = runner.TestLoader()
        self.output = StringIO()
        self.reporter: reporter.Reporter = reporter.Reporter(self.output)
        self.result = UncleanWarningsReporterWrapper(self.reporter)

    def getResult(self, suite: PyUnitTestSuite) -> reporter.Reporter:
        suite.run(self.result)
        return self.reporter