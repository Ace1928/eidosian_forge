from __future__ import annotations as _annotations
import types
import typing
from typing import Any
import typing_extensions
from . import _typing_extra
def __repr_str__(self, join_str: str) -> str:
    return join_str.join((repr(v) if a is None else f'{a}={v!r}' for a, v in self.__repr_args__()))