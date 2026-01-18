from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class GeneratorSchema(TypedDict, total=False):
    type: Required[Literal['generator']]
    items_schema: CoreSchema
    min_length: int
    max_length: int
    ref: str
    metadata: Any
    serialization: IncExSeqOrElseSerSchema