from __future__ import annotations
import re
import sys
import warnings
from functools import wraps, lru_cache
from itertools import count
from typing import TYPE_CHECKING, Generic, Iterator, NamedTuple, TypeVar, TypedDict, overload
class _PriorityItem(NamedTuple):
    name: str
    priority: float