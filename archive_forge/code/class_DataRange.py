from __future__ import annotations
import logging # isort:skip
from collections import Counter
from math import nan
from ..core.enums import PaddingUnits, StartEnd
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import DUPLICATE_FACTORS
from ..model import Model
@abstract
class DataRange(NumericalRange):
    """ A base class for all data range types.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    renderers = Either(List(Instance(Model)), Auto, help='\n    An explicit list of renderers to autorange against. If unset,\n    defaults to all renderers on a plot.\n    ')
    start = Override(default=nan)
    end = Override(default=nan)