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
class CBytes(Bytes, TraitType[bytes, t.Any]):
    """A casting version of the byte string trait."""

    def validate(self, obj: t.Any, value: t.Any) -> bytes | None:
        try:
            return bytes(value)
        except Exception:
            self.error(obj, value)