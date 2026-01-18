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
@abstract
class NumericalSlider(AbstractSlider):
    """ Base class for numerical sliders. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    format = Either(String, Instance(TickFormatter), help='\n    ')