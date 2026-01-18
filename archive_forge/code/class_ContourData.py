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
class ContourData:
    """ Complete geometry data for filled polygons and/or contour lines over a
    whole sequence of contour levels.

    :func:`~bokeh.plotting.contour.contour_data` returns an object of
    this class that can then be passed to :func:`bokeh.models.ContourRenderer.set_data`.
    """
    fill_data: FillData | None
    line_data: LineData | None