from __future__ import annotations
import copyreg
from .pretty import pretty
from typing import Any, Iterator, Hashable, Pattern, Iterable, Mapping
class CustomSelectors(ImmutableDict):
    """Custom selectors."""

    def __init__(self, arg: dict[str, str] | Iterable[tuple[str, str]]) -> None:
        """Initialize."""
        super().__init__(arg)

    def _validate(self, arg: dict[str, str] | Iterable[tuple[str, str]]) -> None:
        """Validate arguments."""
        if isinstance(arg, dict):
            if not all((isinstance(v, str) for v in arg.values())):
                raise TypeError(f'{self.__class__.__name__} values must be hashable')
        elif not all((isinstance(k, str) and isinstance(v, str) for k, v in arg)):
            raise TypeError(f'{self.__class__.__name__} keys and values must be Unicode strings')