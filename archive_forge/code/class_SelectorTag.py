from __future__ import annotations
import copyreg
from .pretty import pretty
from typing import Any, Iterator, Hashable, Pattern, Iterable, Mapping
class SelectorTag(Immutable):
    """Selector tag."""
    __slots__ = ('name', 'prefix', '_hash')
    name: str
    prefix: str | None

    def __init__(self, name: str, prefix: str | None) -> None:
        """Initialize."""
        super().__init__(name=name, prefix=prefix)