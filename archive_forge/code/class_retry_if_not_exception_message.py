import abc
import re
import typing
class retry_if_not_exception_message(retry_if_exception_message):
    """Retries until an exception message equals or matches."""

    def __init__(self, message: typing.Optional[str]=None, match: typing.Optional[str]=None) -> None:
        super().__init__(message, match)
        if_predicate = self.predicate
        self.predicate = lambda *args_, **kwargs_: not if_predicate(*args_, **kwargs_)

    def __call__(self, retry_state: 'RetryCallState') -> bool:
        if retry_state.outcome is None:
            raise RuntimeError('__call__() called before outcome was set')
        if not retry_state.outcome.failed:
            return True
        exception = retry_state.outcome.exception()
        if exception is None:
            raise RuntimeError('outcome failed but the exception is None')
        return self.predicate(exception)