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
class TestCopyStreamResultCopies(TestCase):

    def setUp(self):
        super().setUp()
        self.target1 = LoggingStreamResult()
        self.target2 = LoggingStreamResult()
        self.targets = [self.target1._events, self.target2._events]
        self.result = CopyStreamResult([self.target1, self.target2])

    def test_startTestRun(self):
        self.result.startTestRun()
        self.assertThat(self.targets, AllMatch(Equals([('startTestRun',)])))

    def test_stopTestRun(self):
        self.result.startTestRun()
        self.result.stopTestRun()
        self.assertThat(self.targets, AllMatch(Equals([('startTestRun',), ('stopTestRun',)])))

    def test_status(self):
        self.result.startTestRun()
        now = datetime.datetime.now(utc)
        self.result.status('foo', 'success', test_tags={'tag'}, runnable=False, file_name='foo', file_bytes=b'bar', eof=True, mime_type='text/json', route_code='abc', timestamp=now)
        self.assertThat(self.targets, AllMatch(Equals([('startTestRun',), ('status', 'foo', 'success', {'tag'}, False, 'foo', b'bar', True, 'text/json', 'abc', now)])))