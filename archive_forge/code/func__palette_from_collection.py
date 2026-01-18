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
def _palette_from_collection(collection: PaletteCollection, n: int) -> Palette:
    if len(collection) < 1:
        raise ValueError('PaletteCollection is empty')
    palette = collection.get(n, None)
    if palette is not None:
        return palette
    max_key = max(collection.keys())
    if isinstance(max_key, int) and n > max_key:
        return interp_palette(collection[max_key], n)
    min_key = min(collection.keys())
    if isinstance(min_key, int) and n < min_key:
        return interp_palette(collection[min_key], n)
    raise ValueError(f'Unable to extract or interpolate palette of length {n} from PaletteCollection')