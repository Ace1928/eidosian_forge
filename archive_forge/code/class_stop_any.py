import abc
import typing
from pip._vendor.tenacity import _utils
class stop_any(stop_base):
    """Stop if any of the stop condition is valid."""

    def __init__(self, *stops: stop_base) -> None:
        self.stops = stops

    def __call__(self, retry_state: 'RetryCallState') -> bool:
        return any((x(retry_state) for x in self.stops))