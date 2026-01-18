from testscenarios import multiply_scenarios
from testtools import TestCase
from testtools.matchers import (
def _make_behavior_scenarios(stage):
    """Given a test stage, iterate over behavior scenarios for that stage.

    e.g.
        >>> list(_make_behavior_scenarios('set_up'))
        [('set_up=success', {'set_up_behavior': <function _success>}),
         ('set_up=fail', {'set_up_behavior': <function _failure>}),
         ('set_up=error', {'set_up_behavior': <function _error>}),
         ('set_up=skip', {'set_up_behavior': <function _skip>}),
         ('set_up=xfail', {'set_up_behavior': <function _expected_failure>),
         ('set_up=uxsuccess',
          {'set_up_behavior': <function _unexpected_success>})]

    Ordering is not consistent.
    """
    return ((f'{stage}={behavior}', {f'{stage}_behavior': function}) for behavior, function in behaviors)