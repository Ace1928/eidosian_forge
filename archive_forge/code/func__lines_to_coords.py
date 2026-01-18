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
def _lines_to_coords(lines: LineReturn_ChunkCombinedNan) -> SingleLineCoords:
    points = lines[0][0]
    if points is None:
        empty = np.empty(0)
        return SingleLineCoords(empty, empty)
    xs = points[:, 0]
    ys = points[:, 1]
    return SingleLineCoords(xs, ys)