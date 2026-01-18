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
class FontSizeSpec(DataSpec):
    """ A |DataSpec| property that accepts font-size fixed values.

    The ``FontSizeSpec`` property attempts to first interpret string values as
    font sizes (i.e. valid CSS length values). Otherwise string values are
    interpreted as field names. For example:

    .. code-block:: python

        m.font_size = "13px"  # value

        m.font_size = "1.5em" # value

        m.font_size = "foo"   # field

    A full list of all valid CSS length units can be found here:

    https://drafts.csswg.org/css-values/#lengths

    """

    def __init__(self, default, *, help: str | None=None) -> None:
        super().__init__(FontSize, default=default, help=help)

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        if isinstance(value, str):
            if len(value) == 0 or (value[0].isdigit() and (not FontSize._font_size_re.match(value))):
                msg = '' if not detail else f'{value!r} is not a valid font size value'
                raise ValueError(msg)