from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Callable
from prompt_toolkit.eventloop import run_in_executor_with_context
from .document import Document
from .filters import FilterOrBool, to_filter
class _ValidatorFromCallable(Validator):
    """
    Validate input from a simple callable.
    """

    def __init__(self, func: Callable[[str], bool], error_message: str, move_cursor_to_end: bool) -> None:
        self.func = func
        self.error_message = error_message
        self.move_cursor_to_end = move_cursor_to_end

    def __repr__(self) -> str:
        return f'Validator.from_callable({self.func!r})'

    def validate(self, document: Document) -> None:
        if not self.func(document.text):
            if self.move_cursor_to_end:
                index = len(document.text)
            else:
                index = 0
            raise ValidationError(cursor_position=index, message=self.error_message)