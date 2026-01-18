import sys
from os import PathLike
from typing import TYPE_CHECKING
from typing import List, Dict, AnyStr, Any, Set
from typing import Optional, Union, Tuple, Mapping, Sequence, TypeVar, Type
from typing import Callable, Coroutine, Generator, AsyncGenerator, IO, Iterable, Iterator, AsyncIterator
from typing import cast, overload
from enum import Enum, EnumMeta
from functools import singledispatchmethod
class UpperStrEnum(StrEnum):
    """
    UpperStrEnum is a string enum that allows for case-insensitive comparisons
    """

    def __eq__(self, other: Any) -> bool:
        return self.value.upper() == other.upper() if isinstance(other, str) else super().__eq__(other)

    def __ne__(self, other: Any) -> bool:
        return self.value.upper() != other.upper() if isinstance(other, str) else super().__ne__(other)

    def __str__(self) -> str:
        return str.__str__(self)

    def __hash__(self) -> int:
        return id(self)