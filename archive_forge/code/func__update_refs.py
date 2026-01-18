from __future__ import annotations
import asyncio
import inspect
import math
import operator
from collections.abc import Iterable, Iterator
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Callable, Optional
from .depends import depends
from .display import _display_accessors, _reactive_display_objs
from .parameterized import (
from .parameters import Boolean, Event
from ._utils import _to_async_gen, iscoroutinefunction, full_groupby
def _update_refs(self, refs):
    for w in self._watchers:
        (w.inst or w.cls).param.unwatch(w)
    self._watchers = []
    for _, params in full_groupby(refs, lambda x: id(x.owner)):
        self._watchers.append(params[0].owner.param.watch(self._resolve_value, [p.name for p in params]))