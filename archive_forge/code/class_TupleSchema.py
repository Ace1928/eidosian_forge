from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class TupleSchema(TypedDict, total=False):
    type: Required[Literal['tuple']]
    items_schema: Required[List[CoreSchema]]
    variadic_item_index: int
    min_length: int
    max_length: int
    strict: bool
    ref: str
    metadata: Any
    serialization: IncExSeqOrElseSerSchema