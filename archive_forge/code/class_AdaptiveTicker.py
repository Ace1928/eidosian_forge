from __future__ import annotations
import logging # isort:skip
from ..core.enums import LatLon
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from .mappers import ScanningColorMapper
class AdaptiveTicker(ContinuousTicker):
    """ Generate "nice" round ticks at any magnitude.

    Creates ticks that are "base" multiples of a set of given
    mantissas. For example, with ``base=10`` and
    ``mantissas=[1, 2, 5]``, the ticker will generate the sequence::

        ..., 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, ...

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    base = Float(10.0, help='\n    The multiplier to use for scaling mantissas.\n    ')
    mantissas = Seq(Float, default=[1, 2, 5], help='\n    The acceptable list numbers to generate multiples of.\n    ')
    min_interval = Float(0.0, help='\n    The smallest allowable interval between two adjacent ticks.\n    ')
    max_interval = Nullable(Float, help='\n    The largest allowable interval between two adjacent ticks.\n\n    .. note::\n        To specify an unbounded interval, set to ``None``.\n    ')