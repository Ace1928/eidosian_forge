import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
class StepTracker(object):

    def __init__(self):
        self.expected_groups = []
        self.side_effects = {}

    def expect_call_group(self, step_num, targets):
        self.expected_groups.append(set(((step_num, t) for t in targets)))

    def expect_call(self, step_num, target):
        self.expect_call_group(step_num, (target,))

    def raise_on(self, step_num, target, exc):
        self.side_effects[step_num, target] = exc

    def sleep_on(self, step_num, target, sleep_time):
        self.side_effects[step_num, target] = float(sleep_time)

    def side_effect(self, step_num, target):
        se = self.side_effects.get((step_num, target), None)
        if isinstance(se, BaseException):
            raise se
        if isinstance(se, float):
            eventlet.sleep(se)

    def verify_calls(self, mock_step):
        actual_calls = mock_step.mock_calls
        CallList = type(actual_calls)
        idx = 0
        try:
            for group in self.expected_groups:
                group_len = len(group)
                group_actual = CallList(actual_calls[idx:idx + group_len])
                idx += group_len
                mock_step.mock_calls = group_actual
                mock_step.assert_has_calls([mock.call(s, t) for s, t in group], any_order=True)
        finally:
            mock_step.actual_calls = actual_calls
        if len(actual_calls) > idx:
            raise AssertionError('Unexpected calls: %s' % CallList(actual_calls[idx:]))