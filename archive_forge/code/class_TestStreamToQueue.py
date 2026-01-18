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
class TestStreamToQueue(TestCase):

    def make_result(self):
        queue = Queue()
        return (queue, StreamToQueue(queue, 'foo'))

    def test_status(self):

        def check_event(event_dict, route=None, time=None):
            self.assertEqual('status', event_dict['event'])
            self.assertEqual('test', event_dict['test_id'])
            self.assertEqual('fail', event_dict['test_status'])
            self.assertEqual({'quux'}, event_dict['test_tags'])
            self.assertEqual(False, event_dict['runnable'])
            self.assertEqual('file', event_dict['file_name'])
            self.assertEqual(_b('content'), event_dict['file_bytes'])
            self.assertEqual(True, event_dict['eof'])
            self.assertEqual('quux', event_dict['mime_type'])
            self.assertEqual('test', event_dict['test_id'])
            self.assertEqual(route, event_dict['route_code'])
            self.assertEqual(time, event_dict['timestamp'])
        queue, result = self.make_result()
        result.status('test', 'fail', test_tags={'quux'}, runnable=False, file_name='file', file_bytes=_b('content'), eof=True, mime_type='quux', route_code=None, timestamp=None)
        self.assertEqual(1, queue.qsize())
        a_time = datetime.datetime.now(utc)
        result.status('test', 'fail', test_tags={'quux'}, runnable=False, file_name='file', file_bytes=_b('content'), eof=True, mime_type='quux', route_code='bar', timestamp=a_time)
        self.assertEqual(2, queue.qsize())
        check_event(queue.get(False), route='foo', time=None)
        check_event(queue.get(False), route='foo/bar', time=a_time)

    def testStartTestRun(self):
        queue, result = self.make_result()
        result.startTestRun()
        self.assertEqual({'event': 'startTestRun', 'result': result}, queue.get(False))
        self.assertTrue(queue.empty())

    def testStopTestRun(self):
        queue, result = self.make_result()
        result.stopTestRun()
        self.assertEqual({'event': 'stopTestRun', 'result': result}, queue.get(False))
        self.assertTrue(queue.empty())