from typing import Any, NamedTuple, TYPE_CHECKING
class FormattedSourceLocation(TypedDict):
    """Formatted source location"""
    line: int
    column: int