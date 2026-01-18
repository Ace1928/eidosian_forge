from __future__ import annotations
import logging # isort:skip
from collections.abc import (
from typing import TYPE_CHECKING, Any, TypeVar
from ._sphinx import property_link, register_type_link, type_link
from .bases import (
from .descriptors import ColumnDataPropertyDescriptor
from .enum import Enum
from .numeric import Int
from .singletons import Intrinsic, Undefined
from .wrappers import (
@classmethod
def _is_seq_like(cls, value: Any) -> bool:
    return isinstance(value, (Container, Sized, Iterable)) and hasattr(value, '__getitem__') and (not isinstance(value, Mapping))