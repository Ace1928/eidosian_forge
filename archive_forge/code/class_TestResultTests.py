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
class TestResultTests(unittest.SynchronousTestCase):

    def setUp(self):
        self.result = reporter.TestResult()

    def test_pyunitAddError(self):
        try:
            raise RuntimeError('foo')
        except RuntimeError as e:
            excValue = e
            self.result.addError(self, sys.exc_info())
        failure = self.result.errors[0][1]
        self.assertEqual(excValue, failure.value)
        self.assertEqual(RuntimeError, failure.type)

    def test_pyunitAddFailure(self):
        try:
            raise self.failureException('foo')
        except self.failureException as e:
            excValue = e
            self.result.addFailure(self, sys.exc_info())
        failure = self.result.failures[0][1]
        self.assertEqual(excValue, failure.value)
        self.assertEqual(self.failureException, failure.type)

    def test_somethingElse(self):
        """
        L{reporter.TestResult.addError} raises L{TypeError} if it is called with
        an error that is neither a L{sys.exc_info}-like three-tuple nor a
        L{Failure}.
        """
        with self.assertRaises(TypeError):
            self.result.addError(self, 'an error')
        with self.assertRaises(TypeError):
            self.result.addError(self, Exception('an error'))
        with self.assertRaises(TypeError):
            self.result.addError(self, (Exception, Exception('an error'), None, 'extra'))
        with self.assertRaises(TypeError):
            self.result.addError(self, (Exception, Exception('an error')))