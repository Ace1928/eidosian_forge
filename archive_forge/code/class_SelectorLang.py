from __future__ import annotations
import copyreg
from .pretty import pretty
from typing import Any, Iterator, Hashable, Pattern, Iterable, Mapping
class SelectorLang(Immutable):
    """Selector language rules."""
    __slots__ = ('languages', '_hash')
    languages: tuple[str, ...]

    def __init__(self, languages: Iterable[str]):
        """Initialize."""
        super().__init__(languages=tuple(languages))

    def __iter__(self) -> Iterator[str]:
        """Iterator."""
        return iter(self.languages)

    def __len__(self) -> int:
        """Length."""
        return len(self.languages)

    def __getitem__(self, index: int) -> str:
        """Get item."""
        return self.languages[index]