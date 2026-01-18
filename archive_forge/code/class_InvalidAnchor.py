from __future__ import annotations
from typing import TYPE_CHECKING, Any
import attrs
from referencing._attrs import frozen
@frozen
class InvalidAnchor(Unresolvable):
    """
    An anchor which could never exist in a resource was dereferenced.

    It is somehow syntactically invalid.
    """
    resource: Resource[Any]
    anchor: str

    def __str__(self) -> str:
        return f"'#{self.anchor}' is not a valid anchor, neither as a plain name anchor nor as a JSON Pointer. You may have intended to use '#/{self.anchor}', as the slash is required *before each segment* of a JSON pointer."