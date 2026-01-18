from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class NullableSchema(TypedDict, total=False):
    type: Required[Literal['nullable']]
    schema: Required[CoreSchema]
    strict: bool
    ref: str
    metadata: Any
    serialization: SerSchema