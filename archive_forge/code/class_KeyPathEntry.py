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
class KeyPathEntry(NamedTuple):
    key: Any

    def __add__(self, other: object) -> KeyPath:
        if isinstance(other, KeyPathEntry):
            return KeyPath((self, other))
        if isinstance(other, KeyPath):
            return KeyPath((self, *other.keys))
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.key == other.key

    def pprint(self) -> str:
        """Pretty name of the key path entry."""
        raise NotImplementedError