import abc
import re
import typing
class retry_unless_exception_type(retry_if_exception):
    """Retries until an exception is raised of one or more types."""

    def __init__(self, exception_types: typing.Union[typing.Type[BaseException], typing.Tuple[typing.Type[BaseException], ...]]=Exception) -> None:
        self.exception_types = exception_types
        super().__init__(lambda e: not isinstance(e, exception_types))

    def __call__(self, retry_state: 'RetryCallState') -> bool:
        if retry_state.outcome is None:
            raise RuntimeError('__call__() called before outcome was set')
        if not retry_state.outcome.failed:
            return True
        exception = retry_state.outcome.exception()
        if exception is None:
            raise RuntimeError('outcome failed but the exception is None')
        return self.predicate(exception)