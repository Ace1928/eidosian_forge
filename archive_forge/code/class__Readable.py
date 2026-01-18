from collections.abc import Callable, Iterable, Iterator
from types import TracebackType
from typing import Any, Protocol, TypeAlias
class _Readable(Protocol):

    def read(self, size: int=..., /) -> bytes:
        ...