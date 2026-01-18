from breezy.tests import TestCase, TestLoader, iter_suite_tests, multiply_tests
from breezy.tests.scenarios import (load_tests_apply_scenarios,
class TestTestScenarios(TestCase):

    def test_multiply_tests(self):
        loader = TestLoader()
        suite = loader.suiteClass()
        multiply_tests(self, vary_by_color(), suite)
        self.assertEqual(['blue', 'green', 'red'], get_generated_test_attributes(suite, 'color'))

    def test_multiply_scenarios_from_generators(self):
        """It's safe to multiply scenarios that come from generators"""
        s = multiply_scenarios(vary_named_attribute('one'), vary_named_attribute('two'))
        self.assertEqual(2 * 2, len(s), s)

    def test_multiply_tests_by_their_scenarios(self):
        loader = TestLoader()
        suite = loader.suiteClass()
        test_instance = PretendVaryingTest('test_nothing')
        multiply_tests_by_their_scenarios(test_instance, suite)
        self.assertEqual(['a', 'a', 'b', 'b'], get_generated_test_attributes(suite, 'value'))

    def test_multiply_tests_no_scenarios(self):
        """Tests with no scenarios attribute aren't multiplied"""
        suite = TestLoader().suiteClass()
        multiply_tests_by_their_scenarios(self, suite)
        self.assertLength(1, list(iter_suite_tests(suite)))