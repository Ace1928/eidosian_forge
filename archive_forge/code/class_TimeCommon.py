from __future__ import annotations
import logging # isort:skip
from ...core.enums import CalendarPosition
from ...core.has_props import HasProps, abstract
from ...core.properties import (
from .inputs import InputWidget
@abstract
class TimeCommon(HasProps):
    """ Common properties for time-like picker widgets. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    hour_increment = Positive(Int)(default=1, help='\n    Defines the granularity of hour value incremements in the UI.\n    ')
    minute_increment = Positive(Int)(default=1, help='\n    Defines the granularity of minute value incremements in the UI.\n    ')
    second_increment = Positive(Int)(default=1, help='\n    Defines the granularity of second value incremements in the UI.\n    ')
    seconds = Bool(default=False, help='\n    Allows to select seconds. By default only hours and minuts are\n    selectable, and AM/PM depending on ``clock`` option.\n    ')
    clock = Enum('12h', '24h', default='24h', help='\n    Whether to use 12 hour or 24 hour clock.\n    ')