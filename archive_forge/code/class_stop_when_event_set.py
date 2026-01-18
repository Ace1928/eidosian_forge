import abc
import typing
from pip._vendor.tenacity import _utils
class stop_when_event_set(stop_base):
    """Stop when the given event is set."""

    def __init__(self, event: 'threading.Event') -> None:
        self.event = event

    def __call__(self, retry_state: 'RetryCallState') -> bool:
        return self.event.is_set()