from __future__ import annotations
import logging # isort:skip
from ..core.enums import LatLon
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from .mappers import ScanningColorMapper
class BasicTicker(AdaptiveTicker):
    """ Generate ticks on a linear scale.

    .. note::
        This class may be renamed to ``LinearTicker`` in the future.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)