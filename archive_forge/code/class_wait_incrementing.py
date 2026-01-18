import abc
import random
import typing
from pip._vendor.tenacity import _utils
class wait_incrementing(wait_base):
    """Wait an incremental amount of time after each attempt.

    Starting at a starting value and incrementing by a value for each attempt
    (and restricting the upper limit to some maximum value).
    """

    def __init__(self, start: _utils.time_unit_type=0, increment: _utils.time_unit_type=100, max: _utils.time_unit_type=_utils.MAX_WAIT) -> None:
        self.start = _utils.to_seconds(start)
        self.increment = _utils.to_seconds(increment)
        self.max = _utils.to_seconds(max)

    def __call__(self, retry_state: 'RetryCallState') -> float:
        result = self.start + self.increment * (retry_state.attempt_number - 1)
        return max(0, min(result, self.max))