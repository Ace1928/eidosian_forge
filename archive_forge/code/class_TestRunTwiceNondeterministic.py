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
class TestRunTwiceNondeterministic(TestCase):
    """Can we run the same test case twice?

    Separate suite for non-deterministic tests, which require more complicated
    assertions and scenarios.
    """
    run_tests_with = FullStackRunTest
    scenarios = nondeterministic_sample_cases_scenarios

    def test_runTwice(self):
        test = self.case
        first_result = ExtendedTestResult()
        test.run(first_result)
        second_result = ExtendedTestResult()
        test.run(second_result)
        self.expectThat(first_result._events, self.expected_first_result)
        self.assertThat(second_result._events, self.expected_second_result)