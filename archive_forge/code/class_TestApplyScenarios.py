import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
class TestApplyScenarios(testtools.TestCase):

    def test_calls_apply_scenario(self):
        self.addCleanup(setattr, testscenarios.scenarios, 'apply_scenario', apply_scenario)
        log = []

        def capture(scenario, test):
            log.append((scenario, test))
        testscenarios.scenarios.apply_scenario = capture
        scenarios = ['foo', 'bar']
        result = list(apply_scenarios(scenarios, 'test'))
        self.assertEqual([('foo', 'test'), ('bar', 'test')], log)

    def test_preserves_scenarios_attribute(self):

        class ReferenceTest(unittest.TestCase):
            scenarios = [('demo', {})]

            def test_pass(self):
                pass
        test = ReferenceTest('test_pass')
        tests = list(apply_scenarios(ReferenceTest.scenarios, test))
        self.assertEqual([('demo', {})], ReferenceTest.scenarios)
        self.assertEqual(ReferenceTest.scenarios, tests[0].scenarios)