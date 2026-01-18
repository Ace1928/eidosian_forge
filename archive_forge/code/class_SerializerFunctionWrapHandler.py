from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class SerializerFunctionWrapHandler(Protocol):

    def __call__(self, __input_value: Any, __index_key: int | str | None=None) -> Any:
        ...