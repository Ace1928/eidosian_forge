from __future__ import annotations
from typing import TYPE_CHECKING, Any
import attrs
from referencing._attrs import frozen
@frozen
class NoSuchAnchor(Unresolvable):
    """
    An anchor does not exist within a particular resource.
    """
    resource: Resource[Any]
    anchor: str

    def __str__(self) -> str:
        return f'{self.anchor!r} does not exist within {self.resource.contents!r}'