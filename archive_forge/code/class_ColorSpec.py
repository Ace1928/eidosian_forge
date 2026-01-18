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
class ColorSpec(DataSpec):
    """ A |DataSpec| property that accepts |Color| fixed values.

    The ``ColorSpec`` property attempts to first interpret string values as
    colors. Otherwise, string values are interpreted as field names. For
    example:

    .. code-block:: python

        m.color = "#a4225f"   # value (hex color string)

        m.color = "firebrick" # value (named CSS color string)

        m.color = "foo"       # field (named "foo")

    This automatic interpretation can be override using the dict format
    directly, or by using the |field| function:

    .. code-block:: python

        m.color = { "field": "firebrick" } # field (named "firebrick")

        m.color = field("firebrick")       # field (named "firebrick")

    """

    def __init__(self, default, *, help: str | None=None) -> None:
        help = f'{help or ''}\n{COLOR_DEFAULT_HELP}'
        super().__init__(Nullable(Color), default=default, help=help)

    @classmethod
    def isconst(cls, val):
        """ Whether the value is a string color literal.

        Checks for a well-formed hexadecimal color value or a named color.

        Args:
            val (str) : the value to check

        Returns:
            True, if the value is a string color literal

        """
        return isinstance(val, str) and (len(val) == 7 and val[0] == '#' or val in enums.NamedColor)

    @classmethod
    def is_color_tuple_shape(cls, val):
        """ Whether the value is the correct shape to be a color tuple

        Checks for a 3 or 4-tuple of numbers

        Args:
            val (str) : the value to check

        Returns:
            True, if the value could be a color tuple

        """
        return isinstance(val, tuple) and len(val) in (3, 4) and all((isinstance(v, (float, int)) for v in val))

    def prepare_value(self, cls, name, value):
        if self.is_color_tuple_shape(value):
            value = tuple((int(v) if i < 3 else v for i, v in enumerate(value)))
        return super().prepare_value(cls, name, value)