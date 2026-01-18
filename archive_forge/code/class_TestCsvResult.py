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
class TestCsvResult(testtools.TestCase):

    def parse_stream(self, stream):
        stream.seek(0)
        reader = csv.reader(stream)
        return list(reader)

    def test_csv_output(self):
        stream = StringIO()
        result = subunit.test_results.CsvResult(stream)
        result._now = iter(range(5)).__next__
        result.startTestRun()
        result.startTest(self)
        result.addSuccess(self)
        result.stopTest(self)
        result.stopTestRun()
        self.assertEqual([['test', 'status', 'start_time', 'stop_time'], [self.id(), 'success', '0', '1']], self.parse_stream(stream))

    def test_just_header_when_no_tests(self):
        stream = StringIO()
        result = subunit.test_results.CsvResult(stream)
        result.startTestRun()
        result.stopTestRun()
        self.assertEqual([['test', 'status', 'start_time', 'stop_time']], self.parse_stream(stream))

    def test_no_output_before_events(self):
        stream = StringIO()
        subunit.test_results.CsvResult(stream)
        self.assertEqual([], self.parse_stream(stream))