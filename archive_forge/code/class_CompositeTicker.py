from __future__ import annotations
import logging # isort:skip
from ..core.enums import LatLon
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from .mappers import ScanningColorMapper
class CompositeTicker(ContinuousTicker):
    """ Combine different tickers at different scales.

    Uses the ``min_interval`` and ``max_interval`` interval attributes
    of the tickers to select the appropriate ticker at different
    scales.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    tickers = Seq(Instance(Ticker), default=[], help='\n    A list of Ticker objects to combine at different scales in order\n    to generate tick values. The supplied tickers should be in order.\n    Specifically, if S comes before T, then it should be the case that::\n\n        S.get_max_interval() < T.get_min_interval()\n\n    ')