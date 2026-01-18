from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
class MutableHeaders(Headers):

    def __setitem__(self, key: str, value: str) -> None:
        """
        Set the header `key` to `value`, removing any duplicate entries.
        Retains insertion order.
        """
        set_key = key.lower().encode('latin-1')
        set_value = value.encode('latin-1')
        found_indexes: 'typing.List[int]' = []
        for idx, (item_key, item_value) in enumerate(self._list):
            if item_key == set_key:
                found_indexes.append(idx)
        for idx in reversed(found_indexes[1:]):
            del self._list[idx]
        if found_indexes:
            idx = found_indexes[0]
            self._list[idx] = (set_key, set_value)
        else:
            self._list.append((set_key, set_value))

    def __delitem__(self, key: str) -> None:
        """
        Remove the header `key`.
        """
        del_key = key.lower().encode('latin-1')
        pop_indexes: 'typing.List[int]' = []
        for idx, (item_key, item_value) in enumerate(self._list):
            if item_key == del_key:
                pop_indexes.append(idx)
        for idx in reversed(pop_indexes):
            del self._list[idx]

    def __ior__(self, other: typing.Mapping[str, str]) -> MutableHeaders:
        if not isinstance(other, typing.Mapping):
            raise TypeError(f'Expected a mapping but got {other.__class__.__name__}')
        self.update(other)
        return self

    def __or__(self, other: typing.Mapping[str, str]) -> MutableHeaders:
        if not isinstance(other, typing.Mapping):
            raise TypeError(f'Expected a mapping but got {other.__class__.__name__}')
        new = self.mutablecopy()
        new.update(other)
        return new

    @property
    def raw(self) -> list[tuple[bytes, bytes]]:
        return self._list

    def setdefault(self, key: str, value: str) -> str:
        """
        If the header `key` does not exist, then set it to `value`.
        Returns the header value.
        """
        set_key = key.lower().encode('latin-1')
        set_value = value.encode('latin-1')
        for idx, (item_key, item_value) in enumerate(self._list):
            if item_key == set_key:
                return item_value.decode('latin-1')
        self._list.append((set_key, set_value))
        return value

    def update(self, other: typing.Mapping[str, str]) -> None:
        for key, val in other.items():
            self[key] = val

    def append(self, key: str, value: str) -> None:
        """
        Append a header, preserving any duplicate entries.
        """
        append_key = key.lower().encode('latin-1')
        append_value = value.encode('latin-1')
        self._list.append((append_key, append_value))

    def add_vary_header(self, vary: str) -> None:
        existing = self.get('vary')
        if existing is not None:
            vary = ', '.join([existing, vary])
        self['vary'] = vary