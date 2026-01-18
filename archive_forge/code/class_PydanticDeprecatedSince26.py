from __future__ import annotations as _annotations
from .version import version_short
class PydanticDeprecatedSince26(PydanticDeprecationWarning):
    """A specific `PydanticDeprecationWarning` subclass defining functionality deprecated since Pydantic 2.6."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args, since=(2, 0), expected_removal=(3, 0))