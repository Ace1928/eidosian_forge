import csv
import datetime
import sys
import unittest
from io import StringIO
import testtools
from testtools import TestCase
from testtools.content import TracebackContent, text_content
from testtools.testresult.doubles import ExtendedTestResult
import subunit
import iso8601
import subunit.test_results
class TestHookedTestResultDecorator(unittest.TestCase):

    def setUp(self):
        terminal = unittest.TestResult()
        asserter = AssertBeforeTestResult(terminal, self)
        self.result = LoggingDecorator(asserter)
        asserter.earlier = self.result
        self.decorated = asserter

    def tearDown(self):
        self.assertEqual(1, self.result._calls)
        self.assertEqual(1, self.decorated._calls)

    def test_startTest(self):
        self.result.startTest(self)

    def test_startTestRun(self):
        self.result.startTestRun()

    def test_stopTest(self):
        self.result.stopTest(self)

    def test_stopTestRun(self):
        self.result.stopTestRun()

    def test_addError(self):
        self.result.addError(self, subunit.RemoteError())

    def test_addError_details(self):
        self.result.addError(self, details={})

    def test_addFailure(self):
        self.result.addFailure(self, subunit.RemoteError())

    def test_addFailure_details(self):
        self.result.addFailure(self, details={})

    def test_addSuccess(self):
        self.result.addSuccess(self)

    def test_addSuccess_details(self):
        self.result.addSuccess(self, details={})

    def test_addSkip(self):
        self.result.addSkip(self, 'foo')

    def test_addSkip_details(self):
        self.result.addSkip(self, details={})

    def test_addExpectedFailure(self):
        self.result.addExpectedFailure(self, subunit.RemoteError())

    def test_addExpectedFailure_details(self):
        self.result.addExpectedFailure(self, details={})

    def test_addUnexpectedSuccess(self):
        self.result.addUnexpectedSuccess(self)

    def test_addUnexpectedSuccess_details(self):
        self.result.addUnexpectedSuccess(self, details={})

    def test_progress(self):
        self.result.progress(1, subunit.PROGRESS_SET)

    def test_wasSuccessful(self):
        self.result.wasSuccessful()

    def test_shouldStop(self):
        self.result.shouldStop

    def test_stop(self):
        self.result.stop()

    def test_time(self):
        self.result.time(None)