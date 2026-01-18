import abc
import random
import typing
from pip._vendor.tenacity import _utils
class wait_random(wait_base):
    """Wait strategy that waits a random amount of time between min/max."""

    def __init__(self, min: _utils.time_unit_type=0, max: _utils.time_unit_type=1) -> None:
        self.wait_random_min = _utils.to_seconds(min)
        self.wait_random_max = _utils.to_seconds(max)

    def __call__(self, retry_state: 'RetryCallState') -> float:
        return self.wait_random_min + random.random() * (self.wait_random_max - self.wait_random_min)