from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
class TestRunTwiceDeterminstic(TestCase):
    """Can we run the same test case twice?"""
    run_tests_with = FullStackRunTest
    scenarios = deterministic_sample_cases_scenarios

    def test_runTwice(self):
        test = make_case_for_behavior_scenario(self)
        first_result = ExtendedTestResult()
        test.run(first_result)
        second_result = ExtendedTestResult()
        test.run(second_result)
        self.assertEqual(first_result._events, second_result._events)