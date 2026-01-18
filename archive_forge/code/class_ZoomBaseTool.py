from __future__ import annotations
import logging # isort:skip
import difflib
import typing as tp
from math import nan
from typing import Literal
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property.struct import Optional
from ..core.validation import error
from ..core.validation.errors import NO_RANGE_TOOL_RANGES
from ..model import Model
from ..util.strings import nice_join
from .annotations import BoxAnnotation, PolyAnnotation, Span
from .callbacks import Callback
from .dom import Template
from .glyphs import (
from .nodes import Node
from .ranges import Range
from .renderers import DataRenderer, GlyphRenderer
from .ui import UIElement
@abstract
class ZoomBaseTool(PlotActionTool):
    """ Abstract base class for zoom action tools. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    renderers = Either(Auto, List(Instance(DataRenderer)), default='auto', help='\n    Restrict zoom to ranges used by the provided data renderers. If ``"auto"``\n    then all ranges provided by the cartesian frame will be used.\n    ')
    dimensions = Enum(Dimensions, default='both', help='\n    Which dimensions the zoom tool is constrained to act in. By default\n    the tool will zoom in any dimension, but can be configured to only\n    zoom horizontally across the width of the plot, or vertically across\n    the height of the plot.\n    ')
    factor = Percent(default=0.1, help='\n    Percentage of the range to zoom for each usage of the tool.\n    ')
    level = NonNegative(Int, default=0, help='\n    When working with composite scales (sub-coordinates), this property\n    allows to configure which set of ranges to scale. The default is to\n    scale top-level (frame) ranges.\n    ')