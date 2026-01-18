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
class AbstractTable(Item, _CustomDict):
    """Common behaviour of both :class:`Table` and :class:`InlineTable`"""

    def __init__(self, value: container.Container, trivia: Trivia):
        Item.__init__(self, trivia)
        self._value = value
        for k, v in self._value.body:
            if k is not None:
                dict.__setitem__(self, k.key, v)

    def unwrap(self) -> dict[str, Any]:
        unwrapped = {}
        for k, v in self.items():
            if isinstance(k, Key):
                k = k.key
            if hasattr(v, 'unwrap'):
                v = v.unwrap()
            unwrapped[k] = v
        return unwrapped

    @property
    def value(self) -> container.Container:
        return self._value

    @overload
    def append(self: AT, key: None, value: Comment | Whitespace) -> AT:
        ...

    @overload
    def append(self: AT, key: Key | str, value: Any) -> AT:
        ...

    def append(self, key, value):
        raise NotImplementedError

    @overload
    def add(self: AT, key: Comment | Whitespace) -> AT:
        ...

    @overload
    def add(self: AT, key: Key | str, value: Any=...) -> AT:
        ...

    def add(self, key, value=None):
        if value is None:
            if not isinstance(key, (Comment, Whitespace)):
                msg = 'Non comment/whitespace items must have an associated key'
                raise ValueError(msg)
            key, value = (None, key)
        return self.append(key, value)

    def remove(self: AT, key: Key | str) -> AT:
        self._value.remove(key)
        if isinstance(key, Key):
            key = key.key
        if key is not None:
            dict.__delitem__(self, key)
        return self

    def setdefault(self, key: Key | str, default: Any) -> Any:
        super().setdefault(key, default)
        return self[key]

    def __str__(self):
        return str(self.value)

    def copy(self: AT) -> AT:
        return copy.copy(self)

    def __repr__(self) -> str:
        return repr(self.value)

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __delitem__(self, key: Key | str) -> None:
        self.remove(key)

    def __getitem__(self, key: Key | str) -> Item:
        return cast(Item, self._value[key])

    def __setitem__(self, key: Key | str, value: Any) -> None:
        if not isinstance(value, Item):
            value = item(value, _parent=self)
        is_replace = key in self
        self._value[key] = value
        if key is not None:
            dict.__setitem__(self, key, value)
        if is_replace:
            return
        m = re.match('(?s)^[^ ]*([ ]+).*$', self._trivia.indent)
        if not m:
            return
        indent = m.group(1)
        if not isinstance(value, Whitespace):
            m = re.match('(?s)^([^ ]*)(.*)$', value.trivia.indent)
            if not m:
                value.trivia.indent = indent
            else:
                value.trivia.indent = m.group(1) + indent + m.group(2)