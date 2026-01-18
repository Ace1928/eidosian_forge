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
class CInt(Int[G, S]):
    """A casting version of the int trait."""
    if t.TYPE_CHECKING:

        @t.overload
        def __init__(self: CInt[int, t.Any], default_value: t.Any | Sentinel=..., allow_none: Literal[False]=..., read_only: bool | None=..., help: str | None=..., config: t.Any | None=..., **kwargs: t.Any) -> None:
            ...

        @t.overload
        def __init__(self: CInt[int | None, t.Any], default_value: t.Any | Sentinel | None=..., allow_none: Literal[True]=..., read_only: bool | None=..., help: str | None=..., config: t.Any | None=..., **kwargs: t.Any) -> None:
            ...

        def __init__(self: CInt[int | None, t.Any], default_value: t.Any | Sentinel | None=..., allow_none: bool=..., read_only: bool | None=..., help: str | None=..., config: t.Any | None=..., **kwargs: t.Any) -> None:
            ...

    def validate(self, obj: t.Any, value: t.Any) -> G:
        try:
            value = int(value)
        except Exception:
            self.error(obj, value)
        return t.cast(G, _validate_bounds(self, obj, value))