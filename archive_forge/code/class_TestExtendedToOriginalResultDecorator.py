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
class TestExtendedToOriginalResultDecorator(TestExtendedToOriginalResultDecoratorBase):

    def test_failfast_py26(self):
        self.make_26_result()
        self.assertEqual(False, self.converter.failfast)
        self.converter.failfast = True
        self.assertFalse(hasattr(self.converter.decorated, 'failfast'))

    def test_failfast_py27(self):
        self.make_27_result()
        self.assertEqual(False, self.converter.failfast)
        self.converter.failfast = True
        self.assertEqual(True, self.converter.decorated.failfast)

    def test_progress_py26(self):
        self.make_26_result()
        self.converter.progress(1, 2)

    def test_progress_py27(self):
        self.make_27_result()
        self.converter.progress(1, 2)

    def test_progress_pyextended(self):
        self.make_extended_result()
        self.converter.progress(1, 2)
        self.assertEqual([('progress', 1, 2)], self.result._events)

    def test_shouldStop(self):
        self.make_26_result()
        self.assertEqual(False, self.converter.shouldStop)
        self.converter.decorated.stop()
        self.assertEqual(True, self.converter.shouldStop)

    def test_startTest_py26(self):
        self.make_26_result()
        self.converter.startTest(self)
        self.assertEqual([('startTest', self)], self.result._events)

    def test_startTest_py27(self):
        self.make_27_result()
        self.converter.startTest(self)
        self.assertEqual([('startTest', self)], self.result._events)

    def test_startTest_pyextended(self):
        self.make_extended_result()
        self.converter.startTest(self)
        self.assertEqual([('startTest', self)], self.result._events)

    def test_startTestRun_py26(self):
        self.make_26_result()
        self.converter.startTestRun()
        self.assertEqual([], self.result._events)

    def test_startTestRun_py27(self):
        self.make_27_result()
        self.converter.startTestRun()
        self.assertEqual([('startTestRun',)], self.result._events)

    def test_startTestRun_pyextended(self):
        self.make_extended_result()
        self.converter.startTestRun()
        self.assertEqual([('startTestRun',)], self.result._events)

    def test_stopTest_py26(self):
        self.make_26_result()
        self.converter.stopTest(self)
        self.assertEqual([('stopTest', self)], self.result._events)

    def test_stopTest_py27(self):
        self.make_27_result()
        self.converter.stopTest(self)
        self.assertEqual([('stopTest', self)], self.result._events)

    def test_stopTest_pyextended(self):
        self.make_extended_result()
        self.converter.stopTest(self)
        self.assertEqual([('stopTest', self)], self.result._events)

    def test_stopTestRun_py26(self):
        self.make_26_result()
        self.converter.stopTestRun()
        self.assertEqual([], self.result._events)

    def test_stopTestRun_py27(self):
        self.make_27_result()
        self.converter.stopTestRun()
        self.assertEqual([('stopTestRun',)], self.result._events)

    def test_stopTestRun_pyextended(self):
        self.make_extended_result()
        self.converter.stopTestRun()
        self.assertEqual([('stopTestRun',)], self.result._events)

    def test_tags_py26(self):
        self.make_26_result()
        self.converter.tags({1}, {2})

    def test_tags_py27(self):
        self.make_27_result()
        self.converter.tags({1}, {2})

    def test_tags_pyextended(self):
        self.make_extended_result()
        self.converter.tags({1}, {2})
        self.assertEqual([('tags', {1}, {2})], self.result._events)

    def test_time_py26(self):
        self.make_26_result()
        self.converter.time(1)

    def test_time_py27(self):
        self.make_27_result()
        self.converter.time(1)

    def test_time_pyextended(self):
        self.make_extended_result()
        self.converter.time(1)
        self.assertEqual([('time', 1)], self.result._events)