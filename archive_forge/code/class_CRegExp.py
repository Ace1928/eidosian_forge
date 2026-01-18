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
class CRegExp(TraitType['re.Pattern[t.Any]', t.Union['re.Pattern[t.Any]', str]]):
    """A casting compiled regular expression trait.

    Accepts both strings and compiled regular expressions. The resulting
    attribute will be a compiled regular expression."""
    info_text = 'a regular expression'

    def validate(self, obj: t.Any, value: t.Any) -> re.Pattern[t.Any] | None:
        try:
            return re.compile(value)
        except Exception:
            self.error(obj, value)