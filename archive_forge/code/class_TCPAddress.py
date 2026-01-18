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
class TCPAddress(TraitType[G, S]):
    """A trait for an (ip, port) tuple.

    This allows for both IPv4 IP addresses as well as hostnames.
    """
    default_value = ('127.0.0.1', 0)
    info_text = 'an (ip, port) tuple'
    if t.TYPE_CHECKING:

        @t.overload
        def __init__(self: TCPAddress[tuple[str, int], tuple[str, int]], default_value: bool | Sentinel=..., allow_none: Literal[False]=..., read_only: bool | None=..., help: str | None=..., config: t.Any=..., **kwargs: t.Any) -> None:
            ...

        @t.overload
        def __init__(self: TCPAddress[tuple[str, int] | None, tuple[str, int] | None], default_value: bool | None | Sentinel=..., allow_none: Literal[True]=..., read_only: bool | None=..., help: str | None=..., config: t.Any=..., **kwargs: t.Any) -> None:
            ...

        def __init__(self: TCPAddress[tuple[str, int] | None, tuple[str, int] | None] | TCPAddress[tuple[str, int], tuple[str, int]], default_value: bool | None | Sentinel=Undefined, allow_none: Literal[True, False]=False, read_only: bool | None=None, help: str | None=None, config: t.Any=None, **kwargs: t.Any) -> None:
            ...

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if isinstance(value, tuple):
            if len(value) == 2:
                if isinstance(value[0], str) and isinstance(value[1], int):
                    port = value[1]
                    if port >= 0 and port <= 65535:
                        return t.cast(G, value)
        self.error(obj, value)

    def from_string(self, s: str) -> G:
        if self.allow_none and s == 'None':
            return t.cast(G, None)
        if ':' not in s:
            raise ValueError('Require `ip:port`, got %r' % s)
        ip, port_str = s.split(':', 1)
        port = int(port_str)
        return t.cast(G, (ip, port))