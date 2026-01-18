from __future__ import annotations
import os
import pickle
import re
import sys
import threading
from collections import abc
from collections.abc import Iterator, Mapping, MutableMapping
from functools import lru_cache
from itertools import chain
from typing import Any
class LocaleDataDict(abc.MutableMapping):
    """Dictionary wrapper that automatically resolves aliases to the actual
    values.
    """

    def __init__(self, data: MutableMapping[str | int | None, Any], base: Mapping[str | int | None, Any] | None=None):
        self._data = data
        if base is None:
            base = data
        self.base = base

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str | int | None]:
        return iter(self._data)

    def __getitem__(self, key: str | int | None) -> Any:
        orig = val = self._data[key]
        if isinstance(val, Alias):
            val = val.resolve(self.base)
        if isinstance(val, tuple):
            alias, others = val
            val = alias.resolve(self.base).copy()
            merge(val, others)
        if isinstance(val, dict):
            val = LocaleDataDict(val, base=self.base)
        if val is not orig:
            self._data[key] = val
        return val

    def __setitem__(self, key: str | int | None, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str | int | None) -> None:
        del self._data[key]

    def copy(self) -> LocaleDataDict:
        return LocaleDataDict(self._data.copy(), base=self.base)