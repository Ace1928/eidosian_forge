from __future__ import annotations
import logging # isort:skip
from typing import Any
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.vectorization import Field
from ...core.property_mixins import (
from ...core.validation import error
from ...core.validation.errors import (
from ...model import Model
from ..formatters import TickFormatter
from ..labeling import LabelingPolicy, NoOverlap
from ..mappers import ColorMapper
from ..ranges import Range
from ..renderers import GlyphRenderer
from ..tickers import FixedTicker, Ticker
from .annotation import Annotation
from .dimensional import Dimensional, MetricLength
class ContourColorBar(BaseColorBar):
    """ Color bar used for contours.

    Supports displaying hatch patterns and line styles that contour plots may
    have as well as the usual fill styles.

    Do not create these objects manually, instead use ``ContourRenderer.color_bar``.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    fill_renderer = Instance(GlyphRenderer, help='\n    Glyph renderer used for filled contour polygons.\n    ')
    line_renderer = Instance(GlyphRenderer, help='\n    Glyph renderer used for contour lines.\n    ')
    levels = Seq(Float, default=[], help='\n    Levels at which the contours are calculated.\n    ')