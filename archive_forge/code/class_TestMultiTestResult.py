import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
class TestMultiTestResult(TestCase):
    """Tests for 'MultiTestResult'."""

    def setUp(self):
        super().setUp()
        self.result1 = LoggingResult([])
        self.result2 = LoggingResult([])
        self.multiResult = MultiTestResult(self.result1, self.result2)

    def assertResultLogsEqual(self, expectedEvents):
        """Assert that our test results have received the expected events."""
        self.assertEqual(expectedEvents, self.result1._events)
        self.assertEqual(expectedEvents, self.result2._events)

    def test_repr(self):
        self.assertEqual('<MultiTestResult ({!r}, {!r})>'.format(ExtendedToOriginalDecorator(self.result1), ExtendedToOriginalDecorator(self.result2)), repr(self.multiResult))

    def test_empty(self):
        self.assertResultLogsEqual([])

    def test_failfast_get(self):
        self.assertEqual(False, self.multiResult.failfast)
        self.result1.failfast = True
        self.assertEqual(True, self.multiResult.failfast)

    def test_failfast_set(self):
        self.multiResult.failfast = True
        self.assertEqual(True, self.result1.failfast)
        self.assertEqual(True, self.result2.failfast)

    def test_shouldStop(self):
        self.assertFalse(self.multiResult.shouldStop)
        self.result2.stop()
        self.assertTrue(self.multiResult.shouldStop)

    def test_startTest(self):
        self.multiResult.startTest(self)
        self.assertResultLogsEqual([('startTest', self)])

    def test_stop(self):
        self.assertFalse(self.multiResult.shouldStop)
        self.multiResult.stop()
        self.assertResultLogsEqual(['stop'])

    def test_stopTest(self):
        self.multiResult.stopTest(self)
        self.assertResultLogsEqual([('stopTest', self)])

    def test_addSkipped(self):
        reason = 'Skipped for some reason'
        self.multiResult.addSkip(self, reason)
        self.assertResultLogsEqual([('addSkip', self, reason)])

    def test_addSuccess(self):
        self.multiResult.addSuccess(self)
        self.assertResultLogsEqual([('addSuccess', self)])

    def test_done(self):
        self.multiResult.done()
        self.assertResultLogsEqual(['done'])

    def test_addFailure(self):
        exc_info = make_exception_info(AssertionError, 'failure')
        self.multiResult.addFailure(self, exc_info)
        self.assertResultLogsEqual([('addFailure', self, exc_info)])

    def test_addError(self):
        exc_info = make_exception_info(RuntimeError, 'error')
        self.multiResult.addError(self, exc_info)
        self.assertResultLogsEqual([('addError', self, exc_info)])

    def test_startTestRun(self):
        self.multiResult.startTestRun()
        self.assertResultLogsEqual(['startTestRun'])

    def test_stopTestRun(self):
        self.multiResult.stopTestRun()
        self.assertResultLogsEqual(['stopTestRun'])

    def test_stopTestRun_returns_results(self):

        class Result(LoggingResult):

            def stopTestRun(self):
                super().stopTestRun()
                return 'foo'
        multi_result = MultiTestResult(Result([]), Result([]))
        result = multi_result.stopTestRun()
        self.assertEqual(('foo', 'foo'), result)

    def test_tags(self):
        added_tags = {'foo', 'bar'}
        removed_tags = {'eggs'}
        self.multiResult.tags(added_tags, removed_tags)
        self.assertResultLogsEqual([('tags', added_tags, removed_tags)])

    def test_time(self):
        self.multiResult.time('foo')
        self.assertResultLogsEqual([('time', 'foo')])