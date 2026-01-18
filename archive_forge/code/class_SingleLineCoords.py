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
class SingleLineCoords:
    """ Coordinates for contour lines at a single contour level.

    The x and y coordinates are stored in a single NumPy array each, with a
    np.nan separating each line.
    """
    xs: np.ndarray
    ys: np.ndarray