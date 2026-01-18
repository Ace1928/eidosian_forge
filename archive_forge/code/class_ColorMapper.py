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
class ColorMapper(Mapper):
    """ Base class for color mapper types.

    """

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1:
            kwargs['palette'] = args[0]
        super().__init__(**kwargs)
    palette = Seq(Color, help='\n    A sequence of colors to use as the target palette for mapping.\n\n    This property can also be set as a ``String``, to the name of any of the\n    palettes shown in :ref:`bokeh.palettes`.\n    ').accepts(Enum(Palette), lambda pal: getattr(palettes, pal))
    nan_color = Color(default='gray', help='\n    Color to be used if data is NaN or otherwise not mappable.\n    ')