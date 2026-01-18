from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Literal, TypedDict
from xarray.core.utils import FrozenDict
def _positive_integer(value: int) -> bool:
    return isinstance(value, int) and value > 0