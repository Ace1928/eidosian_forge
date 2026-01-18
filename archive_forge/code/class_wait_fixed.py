import abc
import random
import typing
from pip._vendor.tenacity import _utils
class wait_fixed(wait_base):
    """Wait strategy that waits a fixed amount of time between each retry."""

    def __init__(self, wait: _utils.time_unit_type) -> None:
        self.wait_fixed = _utils.to_seconds(wait)

    def __call__(self, retry_state: 'RetryCallState') -> float:
        return self.wait_fixed