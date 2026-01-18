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
class Bytes(TraitType[bytes, bytes]):
    """A trait for byte strings."""
    default_value = b''
    info_text = 'a bytes object'

    def validate(self, obj: t.Any, value: t.Any) -> bytes | None:
        if isinstance(value, bytes):
            return value
        self.error(obj, value)

    def from_string(self, s: str) -> bytes | None:
        if self.allow_none and s == 'None':
            return None
        if len(s) >= 3:
            for quote in ('"', "'"):
                if s[:2] == f'b{quote}' and s[-1] == quote:
                    old_s = s
                    s = s[2:-1]
                    warn(f'Supporting extra quotes around Bytes is deprecated in traitlets 5.0. Use {s!r} instead of {old_s!r}.', DeprecationWarning, stacklevel=2)
                    break
        return s.encode('utf8')

    def subclass_init(self, cls: type[t.Any]) -> None:
        pass