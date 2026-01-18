import abc
import re
import typing
class retry_if_exception(retry_base):
    """Retry strategy that retries if an exception verifies a predicate."""

    def __init__(self, predicate: typing.Callable[[BaseException], bool]) -> None:
        self.predicate = predicate

    def __call__(self, retry_state: 'RetryCallState') -> bool:
        if retry_state.outcome is None:
            raise RuntimeError('__call__() called before outcome was set')
        if retry_state.outcome.failed:
            exception = retry_state.outcome.exception()
            if exception is None:
                raise RuntimeError('outcome failed but the exception is None')
            return self.predicate(exception)
        else:
            return False