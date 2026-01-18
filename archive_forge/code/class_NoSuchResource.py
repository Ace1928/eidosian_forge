from __future__ import annotations
from typing import TYPE_CHECKING, Any
import attrs
from referencing._attrs import frozen
@frozen
class NoSuchResource(KeyError):
    """
    The given URI is not present in a registry.

    Unlike most exceptions, this class *is* intended to be publicly
    instantiable and *is* part of the public API of the package.
    """
    ref: URI

    def __eq__(self, other: object) -> bool:
        if self.__class__ is not other.__class__:
            return NotImplemented
        return attrs.astuple(self) == attrs.astuple(other)

    def __hash__(self) -> int:
        return hash(attrs.astuple(self))