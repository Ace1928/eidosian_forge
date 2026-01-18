from breezy.tests import TestCase, TestLoader, iter_suite_tests, multiply_tests
from breezy.tests.scenarios import (load_tests_apply_scenarios,
class PretendVaryingTest(TestCase):
    scenarios = multiply_scenarios(vary_named_attribute('value'), vary_named_attribute('other'))

    def test_nothing(self):
        """This test exists just so it can be multiplied"""
        pass