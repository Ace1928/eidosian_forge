import unittest
from testtools.testcase import clone_test_with_new_id
from testscenarios.scenarios import generate_scenarios
class WithScenarios(object):
    __doc__ = 'A mixin for TestCase with support for declarative scenarios.\n    ' + _doc

    def _get_scenarios(self):
        return getattr(self, 'scenarios', None)

    def countTestCases(self):
        scenarios = self._get_scenarios()
        if not scenarios:
            return 1
        else:
            return len(scenarios)

    def debug(self):
        scenarios = self._get_scenarios()
        if scenarios:
            for test in generate_scenarios(self):
                test.debug()
        else:
            return super(WithScenarios, self).debug()

    def run(self, result=None):
        scenarios = self._get_scenarios()
        if scenarios:
            for test in generate_scenarios(self):
                test.run(result)
            return
        else:
            return super(WithScenarios, self).run(result)