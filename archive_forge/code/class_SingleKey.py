from __future__ import annotations
import abc
import copy
import dataclasses
import math
import re
import string
import sys
from datetime import date
from datetime import datetime
from datetime import time
from datetime import tzinfo
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterable
from typing import Iterator
from typing import Sequence
from typing import TypeVar
from typing import cast
from typing import overload
from tomlkit._compat import PY38
from tomlkit._compat import decode
from tomlkit._types import _CustomDict
from tomlkit._types import _CustomFloat
from tomlkit._types import _CustomInt
from tomlkit._types import _CustomList
from tomlkit._types import wrap_method
from tomlkit._utils import CONTROL_CHARS
from tomlkit._utils import escape_string
from tomlkit.exceptions import InvalidStringError
class SingleKey(Key):
    """A single key"""

    def __init__(self, k: str, t: KeyType | None=None, sep: str | None=None, original: str | None=None) -> None:
        if t is None:
            if not k or any((c not in string.ascii_letters + string.digits + '-' + '_' for c in k)):
                t = KeyType.Basic
            else:
                t = KeyType.Bare
        self.t = t
        if sep is None:
            sep = ' = '
        self.sep = sep
        self.key = k
        if original is None:
            key_str = escape_string(k) if t == KeyType.Basic else k
            original = f'{t.value}{key_str}{t.value}'
        self._original = original
        self._keys = [self]
        self._dotted = False

    @property
    def delimiter(self) -> str:
        """The delimiter: double quote/single quote/none"""
        return self.t.value

    def is_bare(self) -> bool:
        """Check if the key is bare"""
        return self.t == KeyType.Bare

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Key):
            return isinstance(other, SingleKey) and self.key == other.key
        return self.key == other