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
def _remove_notifiers(self, handler: t.Callable[..., t.Any] | None, name: Sentinel | str, type: str | Sentinel) -> None:
    try:
        if handler is None:
            del self._trait_notifiers[name][type]
        else:
            self._trait_notifiers[name][type].remove(handler)
    except KeyError:
        pass