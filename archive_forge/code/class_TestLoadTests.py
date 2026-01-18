import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
class TestLoadTests(testtools.TestCase):

    class SampleTest(unittest.TestCase):

        def test_nothing(self):
            pass
        scenarios = [('a', {}), ('b', {})]

    def test_load_tests_apply_scenarios(self):
        suite = load_tests_apply_scenarios(unittest.TestLoader(), [self.SampleTest('test_nothing')], None)
        result_tests = list(testtools.iterate_tests(suite))
        self.assertEquals(2, len(result_tests), result_tests)

    def test_load_tests_apply_scenarios_old_style(self):
        """Call load_tests in the way used by bzr."""
        suite = load_tests_apply_scenarios([self.SampleTest('test_nothing')], self.__class__.__module__, unittest.TestLoader())
        result_tests = list(testtools.iterate_tests(suite))
        self.assertEquals(2, len(result_tests), result_tests)