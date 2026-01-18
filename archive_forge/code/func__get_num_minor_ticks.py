from __future__ import annotations
import logging # isort:skip
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from ..core.properties import Datetime
from ..core.property.singletons import Intrinsic
from ..models import (
def _get_num_minor_ticks(axis_class: type[Axis], num_minor_ticks: int | Literal['auto'] | None) -> int:
    if isinstance(num_minor_ticks, int):
        if num_minor_ticks <= 1:
            raise ValueError('num_minor_ticks must be > 1')
        return num_minor_ticks
    if num_minor_ticks is None:
        return 0
    if num_minor_ticks == 'auto':
        if axis_class is LogAxis:
            return 10
        return 5