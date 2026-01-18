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
class DateSlider(NumericalSlider):
    """ Slider-based date selection widget. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def value_as_datetime(self) -> datetime | None:
        """ Convenience property to retrieve the value as a datetime object.

        Added in version 2.0
        """
        if self.value is None:
            return None
        if isinstance(self.value, numbers.Number):
            return datetime.fromtimestamp(self.value / 1000, tz=timezone.utc)
        return self.value

    @property
    def value_as_date(self) -> date | None:
        """ Convenience property to retrieve the value as a date object.

        Added in version 2.0
        """
        if self.value is None:
            return None
        if isinstance(self.value, numbers.Number):
            dt = datetime.fromtimestamp(self.value / 1000, tz=timezone.utc)
            return date(*dt.timetuple()[:3])
        return self.value
    value = Required(Datetime, help='\n    Initial or selected value.\n    ')
    value_throttled = Readonly(Required(Datetime), help='\n    Initial or selected value, throttled to report only on mouseup.\n    ')
    start = Required(Datetime, help='\n    The minimum allowable value.\n    ')
    end = Required(Datetime, help='\n    The maximum allowable value.\n    ')
    step = Int(default=1, help='\n    The step between consecutive values, in units of days.\n    ')
    format = Override(default='%d %b %Y')