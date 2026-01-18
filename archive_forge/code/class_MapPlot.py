from __future__ import annotations
import logging # isort:skip
from ..core.enums import MapType
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error, warning
from ..core.validation.errors import (
from ..core.validation.warnings import MISSING_RENDERERS
from ..model import Model
from ..models.ranges import Range1d
from .plots import Plot
@abstract
class MapPlot(Plot):
    """ Abstract base class for map plot models.

    """

    def __init__(self, *args, **kwargs) -> None:
        from ..models.ranges import Range1d
        for r in ('x_range', 'y_range'):
            if r in kwargs and (not isinstance(kwargs.get(r), Range1d)):
                raise ValueError(f'Invalid value for {r!r}, MapPlot ranges may only be Range1d, not data ranges')
        super().__init__(*args, **kwargs)

    @error(INCOMPATIBLE_MAP_RANGE_TYPE)
    def _check_incompatible_map_range_type(self):
        from ..models.ranges import Range1d
        if self.x_range is not None and (not isinstance(self.x_range, Range1d)):
            return '%s.x_range' % str(self)
        if self.y_range is not None and (not isinstance(self.y_range, Range1d)):
            return '%s.y_range' % str(self)