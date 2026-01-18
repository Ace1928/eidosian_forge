from __future__ import annotations
import logging # isort:skip
import numbers
from datetime import date, datetime, timezone
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.descriptors import UnsetValueError
from ...core.property.singletons import Undefined
from ...core.validation import error
from ...core.validation.errors import EQUAL_SLIDER_START_END
from ..formatters import TickFormatter
from .widget import Widget
@error(EQUAL_SLIDER_START_END)
def _check_missing_dimension(self):
    if hasattr(self, 'start') and hasattr(self, 'end'):
        if self.start == self.end:
            return f'{self!s} with title {self.title!s}'