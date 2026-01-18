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
class TestResourcedToStreamDecorator(TestCase):

    def setUp(self):
        super().setUp()
        if testresources is None:
            self.skipTest('Need testresources')

    def test_startMakeResource(self):
        log = LoggingStreamResult()
        result = ResourcedToStreamDecorator(log)
        timestamp = datetime.datetime.utcfromtimestamp(3.476)
        result.startTestRun()
        result.time(timestamp)
        resource = testresources.TestResourceManager()
        result.startMakeResource(resource)
        [_, event] = log._events
        self.assertEqual('testresources.TestResourceManager.make', event.test_id)
        self.assertEqual('inprogress', event.test_status)
        self.assertFalse(event.runnable)
        self.assertEqual(timestamp, event.timestamp)

    def test_startMakeResource_with_custom_id_method(self):
        log = LoggingStreamResult()
        result = ResourcedToStreamDecorator(log)
        resource = testresources.TestResourceManager()
        resource.id = lambda: 'nice.resource'
        result.startTestRun()
        result.startMakeResource(resource)
        self.assertEqual('nice.resource.make', log._events[1].test_id)

    def test_stopMakeResource(self):
        log = LoggingStreamResult()
        result = ResourcedToStreamDecorator(log)
        resource = testresources.TestResourceManager()
        result.startTestRun()
        result.stopMakeResource(resource)
        [_, event] = log._events
        self.assertEqual('testresources.TestResourceManager.make', event.test_id)
        self.assertEqual('success', event.test_status)

    def test_startCleanResource(self):
        log = LoggingStreamResult()
        result = ResourcedToStreamDecorator(log)
        resource = testresources.TestResourceManager()
        result.startTestRun()
        result.startCleanResource(resource)
        [_, event] = log._events
        self.assertEqual('testresources.TestResourceManager.clean', event.test_id)
        self.assertEqual('inprogress', event.test_status)

    def test_stopCleanResource(self):
        log = LoggingStreamResult()
        result = ResourcedToStreamDecorator(log)
        resource = testresources.TestResourceManager()
        result.startTestRun()
        result.stopCleanResource(resource)
        [_, event] = log._events
        self.assertEqual('testresources.TestResourceManager.clean', event.test_id)
        self.assertEqual('success', event.test_status)