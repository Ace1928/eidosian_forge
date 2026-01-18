from __future__ import annotations
import collections
import enum
from functools import update_wrapper
import inspect
import itertools
import operator
import re
import sys
import textwrap
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from . import _collections
from . import compat
from ._has_cy import HAS_CYEXTENSION
from .typing import Literal
from .. import exc
class _hash_limit_string(str):
    """A string subclass that can only be hashed on a maximum amount
    of unique values.

    This is used for warnings so that we can send out parameterized warnings
    without the __warningregistry__ of the module,  or the non-overridable
    "once" registry within warnings.py, overloading memory,


    """
    _hash: int

    def __new__(cls, value: str, num: int, args: Sequence[Any]) -> _hash_limit_string:
        interpolated = value % args + ' (this warning may be suppressed after %d occurrences)' % num
        self = super().__new__(cls, interpolated)
        self._hash = hash('%s_%d' % (value, hash(interpolated) % num))
        return self

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        return hash(self) == hash(other)