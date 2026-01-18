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
class Bool(TraitType[G, S]):
    """A boolean (True, False) trait."""
    default_value = False
    info_text = 'a boolean'
    if t.TYPE_CHECKING:

        @t.overload
        def __init__(self: Bool[bool, bool | int], default_value: bool | Sentinel=..., allow_none: Literal[False]=..., read_only: bool | None=..., help: str | None=..., config: t.Any=..., **kwargs: t.Any) -> None:
            ...

        @t.overload
        def __init__(self: Bool[bool | None, bool | int | None], default_value: bool | Sentinel | None=..., allow_none: Literal[True]=..., read_only: bool | None=..., help: str | None=..., config: t.Any=..., **kwargs: t.Any) -> None:
            ...

        def __init__(self: Bool[bool | None, bool | int | None], default_value: bool | Sentinel | None=..., allow_none: bool=..., read_only: bool | None=..., help: str | None=..., config: t.Any=..., **kwargs: t.Any) -> None:
            ...

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if isinstance(value, bool):
            return t.cast(G, value)
        elif isinstance(value, int):
            if value == 1:
                return t.cast(G, True)
            elif value == 0:
                return t.cast(G, False)
        self.error(obj, value)

    def from_string(self, s: str) -> G:
        if self.allow_none and s == 'None':
            return t.cast(G, None)
        s = s.lower()
        if s in {'true', '1'}:
            return t.cast(G, True)
        elif s in {'false', '0'}:
            return t.cast(G, False)
        else:
            raise ValueError('%r is not 1, 0, true, or false')

    def subclass_init(self, cls: type[t.Any]) -> None:
        pass

    def argcompleter(self, **kwargs: t.Any) -> list[str]:
        """Completion hints for argcomplete"""
        completions = ['true', '1', 'false', '0']
        if self.allow_none:
            completions.append('None')
        return completions