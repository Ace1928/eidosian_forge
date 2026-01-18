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
class ReporterTests(ReporterInterfaceTests):
    """
    Tests for the base L{reporter.Reporter} class.
    """

    def setUp(self):
        ReporterInterfaceTests.setUp(self)
        self._timer = 0
        self.result._getTime = self._getTime

    def _getTime(self):
        self._timer += 1
        return self._timer

    def test_startStop(self):
        self.result.startTest(self.test)
        self.result.stopTest(self.test)
        self.assertTrue(self.result._lastTime > 0)
        self.assertEqual(self.result.testsRun, 1)
        self.assertEqual(self.result.wasSuccessful(), True)

    def test_brokenStream(self):
        """
        Test that the reporter safely writes to its stream.
        """
        result = self.resultFactory(stream=BrokenStream(self.stream))
        result._writeln('Hello')
        self.assertEqual(self.stream.getvalue(), 'Hello\n')
        self.stream.truncate(0)
        self.stream.seek(0)
        result._writeln('Hello %s!', 'World')
        self.assertEqual(self.stream.getvalue(), 'Hello World!\n')

    def test_warning(self):
        """
        L{reporter.Reporter} observes warnings emitted by the Twisted log
        system and writes them to its output stream.
        """
        message = RuntimeWarning('some warning text')
        category = 'exceptions.RuntimeWarning'
        filename = 'path/to/some/file.py'
        lineno = 71
        self.publisher.msg(warning=message, category=category, filename=filename, lineno=lineno)
        self.assertEqual(self.stream.getvalue(), '%s:%d: %s: %s\n' % (filename, lineno, category.split('.')[-1], message))

    def test_duplicateWarningSuppressed(self):
        """
        A warning emitted twice within a single test is only written to the
        stream once.
        """
        self.test_warning()
        self.test_warning()

    def test_warningEmittedForNewTest(self):
        """
        A warning emitted again after a new test has started is written to the
        stream again.
        """
        test = self.__class__('test_warningEmittedForNewTest')
        self.result.startTest(test)
        self.stream.seek(0)
        self.stream.truncate()
        self.test_warning()
        self.stream.seek(0)
        self.stream.truncate()
        self.result.stopTest(test)
        self.result.startTest(test)
        self.stream.seek(0)
        self.stream.truncate()
        self.test_warning()

    def test_stopObserving(self):
        """
        L{reporter.Reporter} stops observing log events when its C{done} method
        is called.
        """
        self.result.done()
        self.stream.seek(0)
        self.stream.truncate()
        self.publisher.msg(warning=RuntimeWarning('some message'), category='exceptions.RuntimeWarning', filename='file/name.py', lineno=17)
        self.assertEqual(self.stream.getvalue(), '')