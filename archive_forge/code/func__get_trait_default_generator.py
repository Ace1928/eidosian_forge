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
def _get_trait_default_generator(self, name: str) -> t.Any:
    """Return default generator for a given trait

        Walk the MRO to resolve the correct default generator according to inheritance.
        """
    method_name = '_%s_default' % name
    if method_name in self.__dict__:
        return getattr(self, method_name)
    if method_name in self.__class__.__dict__:
        return getattr(self.__class__, method_name)
    return self._all_trait_default_generators[name]