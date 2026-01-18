from __future__ import annotations
from typing import TYPE_CHECKING, Any
import attrs
from referencing._attrs import frozen
@frozen
class NoInternalID(Exception):
    """
    A resource has no internal ID, but one is needed.

    E.g. in modern JSON Schema drafts, this is the :kw:`$id` keyword.

    One might be needed if a resource was to-be added to a registry but no
    other URI is available, and the resource doesn't declare its canonical URI.
    """
    resource: Resource[Any]

    def __eq__(self, other: object) -> bool:
        if self.__class__ is not other.__class__:
            return NotImplemented
        return attrs.astuple(self) == attrs.astuple(other)

    def __hash__(self) -> int:
        return hash(attrs.astuple(self))