import abc
import re
import typing
class retry_if_exception_type(retry_if_exception):
    """Retries if an exception has been raised of one or more types."""

    def __init__(self, exception_types: typing.Union[typing.Type[BaseException], typing.Tuple[typing.Type[BaseException], ...]]=Exception) -> None:
        self.exception_types = exception_types
        super().__init__(lambda e: isinstance(e, exception_types))