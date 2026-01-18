from __future__ import annotations
import logging # isort:skip
from ...core.enums import CalendarPosition
from ...core.has_props import HasProps, abstract
from ...core.properties import (
from .inputs import InputWidget
@abstract
class BaseDatePicker(PickerBase, DateCommon):
    """ Bases for various calendar-based date picker widgets.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    min_date = Nullable(Date, default=None, help='\n    Optional earliest allowable date.\n    ')
    max_date = Nullable(Date, default=None, help='\n    Optional latest allowable date.\n    ')