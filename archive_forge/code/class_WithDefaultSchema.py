from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class WithDefaultSchema(TypedDict, total=False):
    type: Required[Literal['default']]
    schema: Required[CoreSchema]
    default: Any
    default_factory: Callable[[], Any]
    on_error: Literal['raise', 'omit', 'default']
    validate_default: bool
    strict: bool
    ref: str
    metadata: Any
    serialization: SerSchema