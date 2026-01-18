from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ...util.dataclasses import Unspecified
from ...util.serialization import convert_datetime_type, convert_timedelta_type
from ...util.strings import nice_join
from .. import enums
from .color import ALPHA_DEFAULT_HELP, COLOR_DEFAULT_HELP, Color
from .datetime import Datetime, TimeDelta
from .descriptors import DataSpecPropertyDescriptor, UnitsSpecPropertyDescriptor
from .either import Either
from .enum import Enum
from .instance import Instance
from .nothing import Nothing
from .nullable import Nullable
from .primitive import (
from .serialized import NotSerialized
from .singletons import Undefined
from .struct import Optional, Struct
from .vectorization import (
from .visual import (
class SizeSpec(NumberSpec):
    """ A |DataSpec| property that accepts non-negative numeric fixed values
    for size values or strings that refer to columns in a
    :class:`~bokeh.models.sources.ColumnDataSource`.
    """

    def prepare_value(self, cls, name, value):
        try:
            if value < 0:
                raise ValueError('Screen sizes must be positive')
        except TypeError:
            pass
        return super().prepare_value(cls, name, value)