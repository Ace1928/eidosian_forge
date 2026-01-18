from __future__ import annotations
import logging # isort:skip
from ...core.enums import CalendarPosition
from ...core.has_props import HasProps, abstract
from ...core.properties import (
from .inputs import InputWidget
class DatetimePicker(BaseDatetimePicker):
    """ Calendar-based date and time picker widget.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    value = Nullable(Datetime, default=None, help='\n    The initial or picked date and time.\n    ')