from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
class Int(TraitType[G, S]):
    """An int trait."""
    default_value = 0
    info_text = 'an int'

    @t.overload
    def __init__(self: Int[int, int], default_value: int | Sentinel=..., allow_none: Literal[False]=..., read_only: bool | None=..., help: str | None=..., config: t.Any | None=..., **kwargs: t.Any) -> None:
        ...

    @t.overload
    def __init__(self: Int[int | None, int | None], default_value: int | Sentinel | None=..., allow_none: Literal[True]=..., read_only: bool | None=..., help: str | None=..., config: t.Any | None=..., **kwargs: t.Any) -> None:
        ...

    def __init__(self, default_value: t.Any=Undefined, allow_none: bool=False, read_only: bool | None=None, help: str | None=None, config: t.Any | None=None, **kwargs: t.Any) -> None:
        self.min = kwargs.pop('min', None)
        self.max = kwargs.pop('max', None)
        super().__init__(default_value=default_value, allow_none=allow_none, read_only=read_only, help=help, config=config, **kwargs)

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if not isinstance(value, int):
            self.error(obj, value)
        return t.cast(G, _validate_bounds(self, obj, value))

    def from_string(self, s: str) -> G:
        if self.allow_none and s == 'None':
            return t.cast(G, None)
        return t.cast(G, int(s))

    def subclass_init(self, cls: type[t.Any]) -> None:
        pass