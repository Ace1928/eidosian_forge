from __future__ import annotations
import dataclasses
import functools
import inspect
import sys
from collections import OrderedDict, defaultdict, deque, namedtuple
from operator import methodcaller
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, Sequence, overload
from typing_extensions import Self  # Python 3.11+
from optree import _C
from optree.typing import (
from optree.utils import safe_zip, total_order_sorted, unzip2
class _HashablePartialShim:
    """Object that delegates :meth:`__call__`, :meth:`__hash__`, and :meth:`__eq__` to another object."""
    func: Callable[..., Any]
    args: tuple[Any, ...]
    keywords: dict[str, Any]

    def __init__(self, partial_func: functools.partial) -> None:
        self.partial_func: functools.partial = partial_func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.partial_func(*args, **kwargs)

    def __hash__(self) -> int:
        return hash(self.partial_func)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _HashablePartialShim):
            return self.partial_func == other.partial_func
        return self.partial_func == other