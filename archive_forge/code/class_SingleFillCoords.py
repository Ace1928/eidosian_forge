from __future__ import annotations
import logging # isort:skip
from typing import (
import numpy as np
from ..core.property_mixins import FillProps, HatchProps, LineProps
from ..models.glyphs import MultiLine, MultiPolygons
from ..models.renderers import ContourRenderer, GlyphRenderer
from ..models.sources import ColumnDataSource
from ..palettes import interp_palette
from ..plotting._renderer import _process_sequence_literals
from ..util.dataclasses import dataclass, entries
@dataclass(frozen=True)
class SingleFillCoords:
    """ Coordinates for filled contour polygons between a lower and upper level.

    The first list contains a list for each polygon. The second list contains
    a separate NumPy array for each boundary of that polygon; the first array
    is always the outer boundary, subsequent arrays are holes.
    """
    xs: list[list[np.ndarray]]
    ys: list[list[np.ndarray]]