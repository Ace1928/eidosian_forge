import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
class TestGenerateScenarios(testtools.TestCase):

    def hook_apply_scenarios(self):
        self.addCleanup(setattr, testscenarios.scenarios, 'apply_scenarios', apply_scenarios)
        log = []

        def capture(scenarios, test):
            log.append((scenarios, test))
            return apply_scenarios(scenarios, test)
        testscenarios.scenarios.apply_scenarios = capture
        return log

    def test_generate_scenarios_preserves_normal_test(self):

        class ReferenceTest(unittest.TestCase):

            def test_pass(self):
                pass
        test = ReferenceTest('test_pass')
        log = self.hook_apply_scenarios()
        self.assertEqual([test], list(generate_scenarios(test)))
        self.assertEqual([], log)

    def test_tests_with_scenarios_calls_apply_scenarios(self):

        class ReferenceTest(unittest.TestCase):
            scenarios = [('demo', {})]

            def test_pass(self):
                pass
        test = ReferenceTest('test_pass')
        log = self.hook_apply_scenarios()
        tests = list(generate_scenarios(test))
        self.expectThat(tests[0].id(), EndsWith('ReferenceTest.test_pass(demo)'))
        self.assertEqual([([('demo', {})], test)], log)

    def test_all_scenarios_yielded(self):

        class ReferenceTest(unittest.TestCase):
            scenarios = [('1', {}), ('2', {})]

            def test_pass(self):
                pass
        test = ReferenceTest('test_pass')
        tests = list(generate_scenarios(test))
        self.expectThat(tests[0].id(), EndsWith('ReferenceTest.test_pass(1)'))
        self.expectThat(tests[1].id(), EndsWith('ReferenceTest.test_pass(2)'))

    def test_scenarios_attribute_cleared(self):

        class ReferenceTest(unittest.TestCase):
            scenarios = [('1', {'foo': 1, 'bar': 2}), ('2', {'foo': 2, 'bar': 4})]

            def test_check_foo(self):
                pass
        test = ReferenceTest('test_check_foo')
        tests = list(generate_scenarios(test))
        for adapted in tests:
            self.assertEqual(None, adapted.scenarios)

    def test_multiple_tests(self):

        class Reference1(unittest.TestCase):
            scenarios = [('1', {}), ('2', {})]

            def test_something(self):
                pass

        class Reference2(unittest.TestCase):
            scenarios = [('3', {}), ('4', {})]

            def test_something(self):
                pass
        suite = unittest.TestSuite()
        suite.addTest(Reference1('test_something'))
        suite.addTest(Reference2('test_something'))
        tests = list(generate_scenarios(suite))
        self.assertEqual(4, len(tests))