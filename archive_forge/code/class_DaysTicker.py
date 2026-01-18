from __future__ import annotations
import logging # isort:skip
from ..core.enums import LatLon
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from .mappers import ScanningColorMapper
class DaysTicker(BaseSingleIntervalTicker):
    """ Generate ticks spaced apart by specific, even multiples of days.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    days = Seq(Int, default=[], help='\n    The intervals of days to use.\n    ')
    num_minor_ticks = Override(default=0)