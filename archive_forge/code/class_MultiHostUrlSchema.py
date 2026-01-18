from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class MultiHostUrlSchema(TypedDict, total=False):
    type: Required[Literal['multi-host-url']]
    max_length: int
    allowed_schemes: List[str]
    host_required: bool
    default_host: str
    default_port: int
    default_path: str
    strict: bool
    ref: str
    metadata: Any
    serialization: SerSchema