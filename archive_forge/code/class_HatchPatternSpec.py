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
class HatchPatternSpec(DataSpec):
    """ A |DataSpec| property that accepts hatch pattern types as fixed values.

    The ``HatchPatternSpec`` property attempts to first interpret string values
    as hatch pattern types. Otherwise string values are interpreted as field
    names. For example:

    .. code-block:: python

        m.font_size = "."    # value

        m.font_size = "ring" # value

        m.font_size = "foo"  # field

    """

    def __init__(self, default, *, help: str | None=None) -> None:
        super().__init__(Nullable(HatchPatternType), default=default, help=help)