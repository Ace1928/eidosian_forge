from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class DataclassField(TypedDict, total=False):
    type: Required[Literal['dataclass-field']]
    name: Required[str]
    schema: Required[CoreSchema]
    kw_only: bool
    init: bool
    init_only: bool
    frozen: bool
    validation_alias: Union[str, List[Union[str, int]], List[List[Union[str, int]]]]
    serialization_alias: str
    serialization_exclude: bool
    metadata: Any