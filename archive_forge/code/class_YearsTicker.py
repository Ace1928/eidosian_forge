from __future__ import annotations
import logging # isort:skip
from ..core.enums import LatLon
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from .mappers import ScanningColorMapper
class YearsTicker(BaseSingleIntervalTicker):
    """ Generate ticks spaced apart even numbers of years.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)