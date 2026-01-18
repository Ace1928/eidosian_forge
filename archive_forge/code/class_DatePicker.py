from __future__ import annotations
import logging # isort:skip
from ...core.enums import CalendarPosition
from ...core.has_props import HasProps, abstract
from ...core.properties import (
from .inputs import InputWidget
class DatePicker(BaseDatePicker):
    """ Calendar-based date picker widget.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    value = Nullable(Date, default=None, help='\n    The initial or picked date.\n    ')