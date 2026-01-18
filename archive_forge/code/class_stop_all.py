import abc
import typing
from pip._vendor.tenacity import _utils
class stop_all(stop_base):
    """Stop if all the stop conditions are valid."""

    def __init__(self, *stops: stop_base) -> None:
        self.stops = stops

    def __call__(self, retry_state: 'RetryCallState') -> bool:
        return all((x(retry_state) for x in self.stops))