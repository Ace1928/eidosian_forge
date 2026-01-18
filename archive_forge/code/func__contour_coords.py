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
def _contour_coords(x: ArrayLike | None, y: ArrayLike | None, z: ArrayLike | np.ma.MaskedArray | None, levels: ArrayLike, want_fill: bool, want_line: bool) -> ContourCoords:
    """
    Return the (xs, ys) coords of filled and/or line contours.
    """
    if not want_fill and (not want_line):
        raise RuntimeError('Neither fill nor line requested in _contour_coords')
    from contourpy import FillType, LineType, contour_generator
    cont_gen = contour_generator(x, y, z, line_type=LineType.ChunkCombinedNan, fill_type=FillType.OuterOffset)
    fill_coords = None
    if want_fill:
        all_xs = []
        all_ys = []
        for i in range(len(levels) - 1):
            filled = cont_gen.filled(levels[i], levels[i + 1])
            filled = cast('FillReturn_OuterOffset', filled)
            coords = _filled_to_coords(filled)
            all_xs.append(coords.xs)
            all_ys.append(coords.ys)
        fill_coords = FillCoords(all_xs, all_ys)
    line_coords = None
    if want_line:
        all_xs = []
        all_ys = []
        for level in levels:
            lines = cont_gen.lines(level)
            lines = cast('LineReturn_ChunkCombinedNan', lines)
            coords = _lines_to_coords(lines)
            all_xs.append(coords.xs)
            all_ys.append(coords.ys)
        line_coords = LineCoords(all_xs, all_ys)
    return ContourCoords(fill_coords, line_coords)