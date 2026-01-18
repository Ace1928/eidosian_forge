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
class FuzzyEnum(Enum[G]):
    """An case-ignoring enum matching choices by unique prefixes/substrings."""
    case_sensitive = False
    substring_matching = False

    def __init__(self: FuzzyEnum[t.Any], values: t.Any, default_value: t.Any=Undefined, case_sensitive: bool=False, substring_matching: bool=False, **kwargs: t.Any) -> None:
        self.case_sensitive = case_sensitive
        self.substring_matching = substring_matching
        super().__init__(values, default_value=default_value, **kwargs)

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if not isinstance(value, str):
            self.error(obj, value)
        conv_func = (lambda c: c) if self.case_sensitive else lambda c: c.lower()
        substring_matching = self.substring_matching
        match_func = (lambda v, c: v in c) if substring_matching else lambda v, c: c.startswith(v)
        value = conv_func(value)
        choices = self.values or []
        matches = [match_func(value, conv_func(c)) for c in choices]
        if sum(matches) == 1:
            for v, m in zip(choices, matches):
                if m:
                    return v
        self.error(obj, value)

    def _info(self, as_rst: bool=False) -> str:
        """Returns a description of the trait."""
        none = ' or %s' % ('`None`' if as_rst else 'None') if self.allow_none else ''
        case = 'sensitive' if self.case_sensitive else 'insensitive'
        substr = 'substring' if self.substring_matching else 'prefix'
        return f'any case-{case} {substr} of {self._choices_str(as_rst)}{none}'

    def info(self) -> str:
        return self._info(as_rst=False)

    def info_rst(self) -> str:
        return self._info(as_rst=True)