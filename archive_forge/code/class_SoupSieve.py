from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
class SoupSieve(ct.Immutable):
    """Compiled Soup Sieve selector matching object."""
    pattern: str
    selectors: ct.SelectorList
    namespaces: ct.Namespaces | None
    custom: dict[str, str]
    flags: int
    __slots__ = ('pattern', 'selectors', 'namespaces', 'custom', 'flags', '_hash')

    def __init__(self, pattern: str, selectors: ct.SelectorList, namespaces: ct.Namespaces | None, custom: ct.CustomSelectors | None, flags: int):
        """Initialize."""
        super().__init__(pattern=pattern, selectors=selectors, namespaces=namespaces, custom=custom, flags=flags)

    def match(self, tag: bs4.Tag) -> bool:
        """Match."""
        return CSSMatch(self.selectors, tag, self.namespaces, self.flags).match(tag)

    def closest(self, tag: bs4.Tag) -> bs4.Tag:
        """Match closest ancestor."""
        return CSSMatch(self.selectors, tag, self.namespaces, self.flags).closest()

    def filter(self, iterable: Iterable[bs4.Tag]) -> list[bs4.Tag]:
        """
        Filter.

        `CSSMatch` can cache certain searches for tags of the same document,
        so if we are given a tag, all tags are from the same document,
        and we can take advantage of the optimization.

        Any other kind of iterable could have tags from different documents or detached tags,
        so for those, we use a new `CSSMatch` for each item in the iterable.
        """
        if CSSMatch.is_tag(iterable):
            return CSSMatch(self.selectors, iterable, self.namespaces, self.flags).filter()
        else:
            return [node for node in iterable if not CSSMatch.is_navigable_string(node) and self.match(node)]

    def select_one(self, tag: bs4.Tag) -> bs4.Tag:
        """Select a single tag."""
        tags = self.select(tag, limit=1)
        return tags[0] if tags else None

    def select(self, tag: bs4.Tag, limit: int=0) -> list[bs4.Tag]:
        """Select the specified tags."""
        return list(self.iselect(tag, limit))

    def iselect(self, tag: bs4.Tag, limit: int=0) -> Iterator[bs4.Tag]:
        """Iterate the specified tags."""
        yield from CSSMatch(self.selectors, tag, self.namespaces, self.flags).select(limit)

    def __repr__(self) -> str:
        """Representation."""
        return f'SoupSieve(pattern={self.pattern!r}, namespaces={self.namespaces!r}, custom={self.custom!r}, flags={self.flags!r})'
    __str__ = __repr__