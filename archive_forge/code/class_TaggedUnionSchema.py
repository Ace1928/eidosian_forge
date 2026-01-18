from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class TaggedUnionSchema(TypedDict, total=False):
    type: Required[Literal['tagged-union']]
    choices: Required[Dict[Hashable, CoreSchema]]
    discriminator: Required[Union[str, List[Union[str, int]], List[List[Union[str, int]]], Callable[[Any], Hashable]]]
    custom_error_type: str
    custom_error_message: str
    custom_error_context: Dict[str, Union[str, int, float]]
    strict: bool
    from_attributes: bool
    ref: str
    metadata: Any
    serialization: SerSchema