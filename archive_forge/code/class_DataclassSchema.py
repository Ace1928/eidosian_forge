from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class DataclassSchema(TypedDict, total=False):
    type: Required[Literal['dataclass']]
    cls: Required[Type[Any]]
    schema: Required[CoreSchema]
    fields: Required[List[str]]
    cls_name: str
    post_init: bool
    revalidate_instances: Literal['always', 'never', 'subclass-instances']
    strict: bool
    frozen: bool
    ref: str
    metadata: Any
    serialization: SerSchema
    slots: bool
    config: CoreConfig