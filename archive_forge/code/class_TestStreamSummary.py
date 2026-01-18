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
class TestStreamSummary(TestCase):

    def test_attributes(self):
        result = StreamSummary()
        result.startTestRun()
        self.assertEqual([], result.failures)
        self.assertEqual([], result.errors)
        self.assertEqual([], result.skipped)
        self.assertEqual([], result.expectedFailures)
        self.assertEqual([], result.unexpectedSuccesses)
        self.assertEqual(0, result.testsRun)

    def test_startTestRun(self):
        result = StreamSummary()
        result.startTestRun()
        result.failures.append('x')
        result.errors.append('x')
        result.skipped.append('x')
        result.expectedFailures.append('x')
        result.unexpectedSuccesses.append('x')
        result.testsRun = 1
        result.startTestRun()
        self.assertEqual([], result.failures)
        self.assertEqual([], result.errors)
        self.assertEqual([], result.skipped)
        self.assertEqual([], result.expectedFailures)
        self.assertEqual([], result.unexpectedSuccesses)
        self.assertEqual(0, result.testsRun)

    def test_wasSuccessful(self):
        result = StreamSummary()
        result.startTestRun()
        self.assertEqual(True, result.wasSuccessful())
        result.failures.append('x')
        self.assertEqual(False, result.wasSuccessful())
        result.startTestRun()
        result.errors.append('x')
        self.assertEqual(False, result.wasSuccessful())
        result.startTestRun()
        result.skipped.append('x')
        self.assertEqual(True, result.wasSuccessful())
        result.startTestRun()
        result.expectedFailures.append('x')
        self.assertEqual(True, result.wasSuccessful())
        result.startTestRun()
        result.unexpectedSuccesses.append('x')
        self.assertEqual(True, result.wasSuccessful())

    def test_stopTestRun(self):
        result = StreamSummary()
        result.startTestRun()
        result.status('foo', 'inprogress')
        result.status('foo', 'success')
        result.status('bar', 'skip')
        result.status('baz', 'exists')
        result.stopTestRun()
        self.assertEqual(True, result.wasSuccessful())
        self.assertEqual(2, result.testsRun)

    def test_stopTestRun_inprogress_test_fails(self):
        result = StreamSummary()
        result.startTestRun()
        result.status('foo', 'inprogress')
        result.stopTestRun()
        self.assertEqual(False, result.wasSuccessful())
        self.assertThat(result.errors, HasLength(1))
        self.assertEqual('foo', result.errors[0][0].id())
        self.assertEqual('Test did not complete', result.errors[0][1])
        result.startTestRun()
        result.status('foo', 'inprogress')
        result.status('foo', 'inprogress', route_code='A')
        result.status('foo', 'success', route_code='A')
        result.stopTestRun()
        self.assertEqual(False, result.wasSuccessful())

    def test_status_skip(self):
        result = StreamSummary()
        result.startTestRun()
        result.status(file_name='reason', file_bytes=_b('Missing dependency'), eof=True, mime_type='text/plain; charset=utf8', test_id='foo.bar')
        result.status('foo.bar', 'skip')
        self.assertThat(result.skipped, HasLength(1))
        self.assertEqual('foo.bar', result.skipped[0][0].id())
        self.assertEqual('Missing dependency', result.skipped[0][1])

    def _report_files(self, result):
        result.status(file_name='some log.txt', file_bytes=_b('1234 log message'), eof=True, mime_type='text/plain; charset=utf8', test_id='foo.bar')
        result.status(file_name='traceback', file_bytes=_b('Traceback (most recent call last):\n  File "testtools/tests/test_testresult.py", line 607, in test_stopTestRun\n      AllMatch(Equals([(\'startTestRun\',), (\'stopTestRun\',)])))\ntesttools.matchers._impl.MismatchError: Differences: [\n[(\'startTestRun\',), (\'stopTestRun\',)] != []\n[(\'startTestRun\',), (\'stopTestRun\',)] != []\n]\n'), eof=True, mime_type='text/plain; charset=utf8', test_id='foo.bar')
    files_message = Equals('some log.txt: {{{1234 log message}}}\n\nTraceback (most recent call last):\n  File "testtools/tests/test_testresult.py", line 607, in test_stopTestRun\n      AllMatch(Equals([(\'startTestRun\',), (\'stopTestRun\',)])))\ntesttools.matchers._impl.MismatchError: Differences: [\n[(\'startTestRun\',), (\'stopTestRun\',)] != []\n[(\'startTestRun\',), (\'stopTestRun\',)] != []\n]\n')

    def test_status_fail(self):
        result = StreamSummary()
        result.startTestRun()
        self._report_files(result)
        result.status('foo.bar', 'fail')
        self.assertThat(result.errors, HasLength(1))
        self.assertEqual('foo.bar', result.errors[0][0].id())
        self.assertThat(result.errors[0][1], self.files_message)

    def test_status_xfail(self):
        result = StreamSummary()
        result.startTestRun()
        self._report_files(result)
        result.status('foo.bar', 'xfail')
        self.assertThat(result.expectedFailures, HasLength(1))
        self.assertEqual('foo.bar', result.expectedFailures[0][0].id())
        self.assertThat(result.expectedFailures[0][1], self.files_message)

    def test_status_uxsuccess(self):
        result = StreamSummary()
        result.startTestRun()
        result.status('foo.bar', 'uxsuccess')
        self.assertThat(result.unexpectedSuccesses, HasLength(1))
        self.assertEqual('foo.bar', result.unexpectedSuccesses[0].id())