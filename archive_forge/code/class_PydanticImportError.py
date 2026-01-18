from __future__ import annotations as _annotations
import re
from typing_extensions import Literal, Self
from ._migration import getattr_migration
from .version import version_short
class PydanticImportError(PydanticErrorMixin, ImportError):
    """An error raised when an import fails due to module changes between V1 and V2.

    Attributes:
        message: Description of the error.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, code='import-error')