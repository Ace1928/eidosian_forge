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
class TestStreamTagger(TestCase):

    def test_adding(self):
        log = LoggingStreamResult()
        result = StreamTagger([log], add=['foo'])
        result.startTestRun()
        result.status()
        result.status(test_tags={'bar'})
        result.status(test_tags=None)
        result.stopTestRun()
        self.assertEqual([('startTestRun',), ('status', None, None, {'foo'}, True, None, None, False, None, None, None), ('status', None, None, {'foo', 'bar'}, True, None, None, False, None, None, None), ('status', None, None, {'foo'}, True, None, None, False, None, None, None), ('stopTestRun',)], log._events)

    def test_discarding(self):
        log = LoggingStreamResult()
        result = StreamTagger([log], discard=['foo'])
        result.startTestRun()
        result.status()
        result.status(test_tags=None)
        result.status(test_tags={'foo'})
        result.status(test_tags={'bar'})
        result.status(test_tags={'foo', 'bar'})
        result.stopTestRun()
        self.assertEqual([('startTestRun',), ('status', None, None, None, True, None, None, False, None, None, None), ('status', None, None, None, True, None, None, False, None, None, None), ('status', None, None, None, True, None, None, False, None, None, None), ('status', None, None, {'bar'}, True, None, None, False, None, None, None), ('status', None, None, {'bar'}, True, None, None, False, None, None, None), ('stopTestRun',)], log._events)