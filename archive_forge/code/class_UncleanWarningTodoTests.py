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
class UncleanWarningTodoTests(TodoTests):
    """
    Tests for L{UncleanWarningsReporterWrapper}'s handling of todos.
    """

    def setUp(self):
        TodoTests.setUp(self)
        self.result = UncleanWarningsReporterWrapper(self.result)

    def _getTodos(self, result):
        """
        Get the  todos that happened to a reporter inside of an unclean
        warnings reporter wrapper.
        """
        return result._originalReporter.expectedFailures

    def _getUnexpectedSuccesses(self, result):
        """
        Get the number of unexpected successes that happened to a reporter
        inside of an unclean warnings reporter wrapper.
        """
        return result._originalReporter.unexpectedSuccesses