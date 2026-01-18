from __future__ import annotations
import copyreg
from .pretty import pretty
from typing import Any, Iterator, Hashable, Pattern, Iterable, Mapping
class SelectorContains(Immutable):
    """Selector contains rule."""
    __slots__ = ('text', 'own', '_hash')
    text: tuple[str, ...]
    own: bool

    def __init__(self, text: Iterable[str], own: bool) -> None:
        """Initialize."""
        super().__init__(text=tuple(text), own=own)