from __future__ import annotations as _annotations
import re
from typing_extensions import Literal, Self
from ._migration import getattr_migration
from .version import version_short
class PydanticErrorMixin:
    """A mixin class for common functionality shared by all Pydantic-specific errors.

    Attributes:
        message: A message describing the error.
        code: An optional error code from PydanticErrorCodes enum.
    """

    def __init__(self, message: str, *, code: PydanticErrorCodes | None) -> None:
        self.message = message
        self.code = code

    def __str__(self) -> str:
        if self.code is None:
            return self.message
        else:
            return f'{self.message}\n\nFor further information visit {DEV_ERROR_DOCS_URL}{self.code}'