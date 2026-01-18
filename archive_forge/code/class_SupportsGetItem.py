from __future__ import annotations
from collections.abc import MutableMapping
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING
class SupportsGetItem(Protocol):
    """
        Supports __getitem__
        """

    def __getitem__(self, key: str, /) -> Any:
        ...

    def __iter__(self) -> Iterator[Hashable]:
        ...