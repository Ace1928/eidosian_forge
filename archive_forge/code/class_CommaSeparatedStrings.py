from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
class CommaSeparatedStrings(typing.Sequence[str]):

    def __init__(self, value: str | typing.Sequence[str]):
        if isinstance(value, str):
            splitter = shlex(value, posix=True)
            splitter.whitespace = ','
            splitter.whitespace_split = True
            self._items = [item.strip() for item in splitter]
        else:
            self._items = list(value)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int | slice) -> typing.Any:
        return self._items[index]

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._items)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        items = [item for item in self]
        return f'{class_name}({items!r})'

    def __str__(self) -> str:
        return ', '.join((repr(item) for item in self))