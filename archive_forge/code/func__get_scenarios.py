import unittest
from testtools.testcase import clone_test_with_new_id
from testscenarios.scenarios import generate_scenarios
def _get_scenarios(self):
    return getattr(self, 'scenarios', None)