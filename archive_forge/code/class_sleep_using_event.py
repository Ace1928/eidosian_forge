import time
import typing
class sleep_using_event:
    """Sleep strategy that waits on an event to be set."""

    def __init__(self, event: 'threading.Event') -> None:
        self.event = event

    def __call__(self, timeout: typing.Optional[float]) -> None:
        self.event.wait(timeout=timeout)