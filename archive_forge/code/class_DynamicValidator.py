from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Callable
from prompt_toolkit.eventloop import run_in_executor_with_context
from .document import Document
from .filters import FilterOrBool, to_filter
class DynamicValidator(Validator):
    """
    Validator class that can dynamically returns any Validator.

    :param get_validator: Callable that returns a :class:`.Validator` instance.
    """

    def __init__(self, get_validator: Callable[[], Validator | None]) -> None:
        self.get_validator = get_validator

    def validate(self, document: Document) -> None:
        validator = self.get_validator() or DummyValidator()
        validator.validate(document)

    async def validate_async(self, document: Document) -> None:
        validator = self.get_validator() or DummyValidator()
        await validator.validate_async(document)