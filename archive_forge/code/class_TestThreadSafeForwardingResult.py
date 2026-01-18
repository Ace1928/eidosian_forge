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
class TestThreadSafeForwardingResult(TestCase):
    """Tests for `TestThreadSafeForwardingResult`."""

    def make_results(self, n):
        events = []
        target = LoggingResult(events)
        semaphore = threading.Semaphore(1)
        return ([ThreadsafeForwardingResult(target, semaphore) for i in range(n)], events)

    def test_nonforwarding_methods(self):
        [result], events = self.make_results(1)
        result.startTest(self)
        result.stopTest(self)
        self.assertEqual([], events)

    def test_tags_not_forwarded(self):
        [result], events = self.make_results(1)
        result.tags({'foo'}, {'bar'})
        self.assertEqual([], events)

    def test_global_tags_simple(self):
        [result], events = self.make_results(1)
        result.tags({'foo'}, set())
        result.time(1)
        result.startTest(self)
        result.time(2)
        result.addSuccess(self)
        self.assertEqual([('time', 1), ('startTest', self), ('time', 2), ('tags', {'foo'}, set()), ('addSuccess', self), ('stopTest', self)], events)

    def test_global_tags_complex(self):
        [result], events = self.make_results(1)
        result.tags({'foo', 'bar'}, {'baz', 'qux'})
        result.tags({'cat', 'qux'}, {'bar', 'dog'})
        result.time(1)
        result.startTest(self)
        result.time(2)
        result.addSuccess(self)
        self.assertEqual([('time', 1), ('startTest', self), ('time', 2), ('tags', {'cat', 'foo', 'qux'}, {'dog', 'bar', 'baz'}), ('addSuccess', self), ('stopTest', self)], events)

    def test_local_tags(self):
        [result], events = self.make_results(1)
        result.time(1)
        result.startTest(self)
        result.tags({'foo'}, set())
        result.tags(set(), {'bar'})
        result.time(2)
        result.addSuccess(self)
        self.assertEqual([('time', 1), ('startTest', self), ('time', 2), ('tags', {'foo'}, {'bar'}), ('addSuccess', self), ('stopTest', self)], events)

    def test_local_tags_dont_leak(self):
        [result], events = self.make_results(1)
        a, b = (PlaceHolder('a'), PlaceHolder('b'))
        result.time(1)
        result.startTest(a)
        result.tags({'foo'}, set())
        result.time(2)
        result.addSuccess(a)
        result.stopTest(a)
        result.time(3)
        result.startTest(b)
        result.time(4)
        result.addSuccess(b)
        result.stopTest(b)
        self.assertEqual([('time', 1), ('startTest', a), ('time', 2), ('tags', {'foo'}, set()), ('addSuccess', a), ('stopTest', a), ('time', 3), ('startTest', b), ('time', 4), ('addSuccess', b), ('stopTest', b)], events)

    def test_startTestRun(self):
        [result1, result2], events = self.make_results(2)
        result1.startTestRun()
        result2.startTestRun()
        self.assertEqual(['startTestRun', 'startTestRun'], events)

    def test_stopTestRun(self):
        [result1, result2], events = self.make_results(2)
        result1.stopTestRun()
        result2.stopTestRun()
        self.assertEqual(['stopTestRun', 'stopTestRun'], events)

    def test_forward_addError(self):
        [result], events = self.make_results(1)
        exc_info = make_exception_info(RuntimeError, 'error')
        start_time = datetime.datetime.utcfromtimestamp(1.489)
        end_time = datetime.datetime.utcfromtimestamp(51.476)
        result.time(start_time)
        result.startTest(self)
        result.time(end_time)
        result.addError(self, exc_info)
        self.assertEqual([('time', start_time), ('startTest', self), ('time', end_time), ('addError', self, exc_info), ('stopTest', self)], events)

    def test_forward_addFailure(self):
        [result], events = self.make_results(1)
        exc_info = make_exception_info(AssertionError, 'failure')
        start_time = datetime.datetime.utcfromtimestamp(2.489)
        end_time = datetime.datetime.utcfromtimestamp(3.476)
        result.time(start_time)
        result.startTest(self)
        result.time(end_time)
        result.addFailure(self, exc_info)
        self.assertEqual([('time', start_time), ('startTest', self), ('time', end_time), ('addFailure', self, exc_info), ('stopTest', self)], events)

    def test_forward_addSkip(self):
        [result], events = self.make_results(1)
        reason = 'Skipped for some reason'
        start_time = datetime.datetime.utcfromtimestamp(4.489)
        end_time = datetime.datetime.utcfromtimestamp(5.476)
        result.time(start_time)
        result.startTest(self)
        result.time(end_time)
        result.addSkip(self, reason)
        self.assertEqual([('time', start_time), ('startTest', self), ('time', end_time), ('addSkip', self, reason), ('stopTest', self)], events)

    def test_forward_addSuccess(self):
        [result], events = self.make_results(1)
        start_time = datetime.datetime.utcfromtimestamp(6.489)
        end_time = datetime.datetime.utcfromtimestamp(7.476)
        result.time(start_time)
        result.startTest(self)
        result.time(end_time)
        result.addSuccess(self)
        self.assertEqual([('time', start_time), ('startTest', self), ('time', end_time), ('addSuccess', self), ('stopTest', self)], events)

    def test_only_one_test_at_a_time(self):
        [result1, result2], events = self.make_results(2)
        test1, test2 = (self, make_test())
        start_time1 = datetime.datetime.utcfromtimestamp(1.489)
        end_time1 = datetime.datetime.utcfromtimestamp(2.476)
        start_time2 = datetime.datetime.utcfromtimestamp(3.489)
        end_time2 = datetime.datetime.utcfromtimestamp(4.489)
        result1.time(start_time1)
        result2.time(start_time2)
        result1.startTest(test1)
        result2.startTest(test2)
        result1.time(end_time1)
        result2.time(end_time2)
        result2.addSuccess(test2)
        result1.addSuccess(test1)
        self.assertEqual([('time', start_time2), ('startTest', test2), ('time', end_time2), ('addSuccess', test2), ('stopTest', test2), ('time', start_time1), ('startTest', test1), ('time', end_time1), ('addSuccess', test1), ('stopTest', test1)], events)