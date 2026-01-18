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
class CategoricalSlider(AbstractSlider):
    """ Discrete slider allowing selection from a collection of values. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    categories = Required(Seq(String), help='\n    A collection of categories to choose from.\n    ')
    value = Required(String, help='\n    Initial or selected value.\n    ')
    value_throttled = Readonly(Required(String), help='\n    Initial or throttled selected value.\n    ')