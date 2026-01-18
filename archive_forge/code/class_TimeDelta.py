from __future__ import annotations
import logging # isort:skip
import datetime
from typing import Any, Union
from ...util.serialization import (
from .bases import Init, Property
from .primitive import bokeh_integer_types
from .singletons import Undefined
class TimeDelta(Property[datetime.timedelta]):
    """ Accept TimeDelta values.

    """

    def __init__(self, default: Init[datetime.timedelta]=datetime.timedelta(), *, help: str | None=None) -> None:
        super().__init__(default=default, help=help)

    def transform(self, value: Any) -> Any:
        value = super().transform(value)
        return value

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        if is_timedelta_type(value):
            return
        msg = '' if not detail else f'Expected a timedelta instance, got {value!r}'
        raise ValueError(msg)