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
class TimeCapturingResult(unittest.TestResult):

    def __init__(self):
        super().__init__()
        self._calls = []
        self.failfast = False

    def time(self, a_datetime):
        self._calls.append(a_datetime)