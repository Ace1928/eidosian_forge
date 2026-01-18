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
class Python27Contract(Python26Contract):

    def test_addExpectedFailure(self):
        result = self.makeResult()
        result.startTest(self)
        result.addExpectedFailure(self, an_exc_info)

    def test_addExpectedFailure_is_success(self):
        result = self.makeResult()
        result.startTest(self)
        result.addExpectedFailure(self, an_exc_info)
        result.stopTest(self)
        self.assertTrue(result.wasSuccessful())

    def test_addSkipped(self):
        result = self.makeResult()
        result.startTest(self)
        result.addSkip(self, 'Skipped for some reason')

    def test_addSkip_is_success(self):
        result = self.makeResult()
        result.startTest(self)
        result.addSkip(self, 'Skipped for some reason')
        result.stopTest(self)
        self.assertTrue(result.wasSuccessful())

    def test_addUnexpectedSuccess(self):
        result = self.makeResult()
        result.startTest(self)
        result.addUnexpectedSuccess(self)

    def test_addUnexpectedSuccess_was_successful(self):
        result = self.makeResult()
        result.startTest(self)
        result.addUnexpectedSuccess(self)
        result.stopTest(self)
        self.assertTrue(result.wasSuccessful())

    def test_startStopTestRun(self):
        result = self.makeResult()
        result.startTestRun()
        result.stopTestRun()

    def test_failfast(self):
        result = self.makeResult()
        result.failfast = True

        class Failing(TestCase):

            def test_a(self):
                self.fail('a')

            def test_b(self):
                self.fail('b')
        TestSuite([Failing('test_a'), Failing('test_b')]).run(result)
        self.assertEqual(1, result.testsRun)