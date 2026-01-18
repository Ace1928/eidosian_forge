from __future__ import annotations
import logging # isort:skip
from .. import palettes
from ..core.enums import Palette
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error, warning
from ..core.validation.errors import WEIGHTED_STACK_COLOR_MAPPER_LABEL_LENGTH_MISMATCH
from ..core.validation.warnings import PALETTE_LENGTH_FACTORS_MISMATCH
from .transforms import Transform
@abstract
class ContinuousColorMapper(ColorMapper):
    """ Base class for continuous color mapper types.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    domain = List(Tuple(Instance('bokeh.models.renderers.GlyphRenderer'), Either(String, List(String))), default=[], help='\n    A collection of glyph renderers to pool data from for establishing data metrics.\n    If empty, mapped data will be used instead.\n    ')
    low = Nullable(Float, help='\n    The minimum value of the range to map into the palette. Values below\n    this are clamped to ``low``. If ``None``, the value is inferred from data.\n    ')
    high = Nullable(Float, help='\n    The maximum value of the range to map into the palette. Values above\n    this are clamped to ``high``. If ``None``, the value is inferred from data.\n    ')
    low_color = Nullable(Color, help='\n    Color to be used if data is lower than ``low`` value. If None,\n    values lower than ``low`` are mapped to the first color in the palette.\n    ')
    high_color = Nullable(Color, help='\n    Color to be used if data is higher than ``high`` value. If None,\n    values higher than ``high`` are mapped to the last color in the palette.\n    ')