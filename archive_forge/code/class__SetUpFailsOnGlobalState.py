from testscenarios import multiply_scenarios
from testtools import TestCase
from testtools.matchers import (
class _SetUpFailsOnGlobalState(TestCase):
    """Fail to upcall setUp on first run. Fail to upcall tearDown after.

    This simulates a test that fails to upcall in ``setUp`` if some global
    state is broken, and fails to call ``tearDown`` when the global state
    breaks but works after that.
    """
    first_run = True

    def setUp(self):
        if not self.first_run:
            return
        super().setUp()

    def test_success(self):
        pass

    def tearDown(self):
        if not self.first_run:
            super().tearDown()
        self.__class__.first_run = False

    @classmethod
    def make_scenario(cls):
        case = cls('test_success')
        return {'case': case, 'expected_first_result': _test_error_traceback(case, Contains('TestCase.tearDown was not called')), 'expected_second_result': _test_error_traceback(case, Contains('TestCase.setUp was not called'))}