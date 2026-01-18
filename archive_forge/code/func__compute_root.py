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
def _compute_root(self):
    if self._prev is None:
        return self
    root = self
    while root._prev is not None:
        root = root._prev
    return root