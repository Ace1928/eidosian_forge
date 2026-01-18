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
def _add_notifiers(self, handler: t.Callable[..., t.Any], name: Sentinel | str, type: str | Sentinel) -> None:
    if name not in self._trait_notifiers:
        nlist: list[t.Any] = []
        self._trait_notifiers[name] = {type: nlist}
    elif type not in self._trait_notifiers[name]:
        nlist = []
        self._trait_notifiers[name][type] = nlist
    else:
        nlist = self._trait_notifiers[name][type]
    if handler not in nlist:
        nlist.append(handler)