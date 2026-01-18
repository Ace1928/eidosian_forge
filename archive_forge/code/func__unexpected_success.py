from testscenarios import multiply_scenarios
from testtools import TestCase
from testtools.matchers import (
def _unexpected_success(case):
    case.expectFailure('arbitrary unexpected success', _success, case)