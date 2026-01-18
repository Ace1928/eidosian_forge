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
class CaselessStrEnum(Enum[G]):
    """An enum of strings where the case should be ignored."""

    def __init__(self: CaselessStrEnum[t.Any], values: t.Any, default_value: t.Any=Undefined, **kwargs: t.Any) -> None:
        super().__init__(values, default_value=default_value, **kwargs)

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if not isinstance(value, str):
            self.error(obj, value)
        for v in self.values or []:
            assert isinstance(v, str)
            if v.lower() == value.lower():
                return t.cast(G, v)
        self.error(obj, value)

    def _info(self, as_rst: bool=False) -> str:
        """Returns a description of the trait."""
        none = ' or %s' % ('`None`' if as_rst else 'None') if self.allow_none else ''
        return f'any of {self._choices_str(as_rst)} (case-insensitive){none}'

    def info(self) -> str:
        return self._info(as_rst=False)

    def info_rst(self) -> str:
        return self._info(as_rst=True)