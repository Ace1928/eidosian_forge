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
class ForwardDeclaredMixin:
    """
    Mixin for forward-declared versions of Instance and Type.
    """

    def _resolve_string(self, string: str) -> t.Any:
        """
        Find the specified class name by looking for it in the module in which
        our this_class attribute was defined.
        """
        modname = self.this_class.__module__
        return import_item('.'.join([modname, string]))