from __future__ import annotations
import logging # isort:skip
from ...core.enums import CalendarPosition
from ...core.has_props import HasProps, abstract
from ...core.properties import (
from .inputs import InputWidget
class MultipleDatePicker(BaseDatePicker):
    """ Calendar-based picker of dates. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    value = List(Date, default=[], help='\n    The initial or picked dates.\n    ')
    separator = String(default=', ', help='\n    The separator between displayed dates.\n    ')