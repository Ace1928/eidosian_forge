from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class DataclassArgsSchema(TypedDict, total=False):
    type: Required[Literal['dataclass-args']]
    dataclass_name: Required[str]
    fields: Required[List[DataclassField]]
    computed_fields: List[ComputedField]
    populate_by_name: bool
    collect_init_only: bool
    ref: str
    metadata: Any
    serialization: SerSchema
    extra_behavior: ExtraBehavior