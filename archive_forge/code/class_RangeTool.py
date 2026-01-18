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
class RangeTool(Tool):
    """ *toolbar icon*: |range_icon|

    The range tool allows the user to update range objects for either or both
    of the x- or y-dimensions by dragging a corresponding shaded annotation to
    move it or change its boundaries.

    A common use case is to add this tool to a plot with a large fixed range,
    but to configure the tool range from a different plot. When the user
    manipulates the overlay, the range of the second plot will be updated
    automatically.

    .. |range_icon| image:: /_images/icons/Range.png
        :height: 24px


    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    x_range = Nullable(Instance(Range), help='\n    A range synchronized to the x-dimension of the overlay. If None, the overlay\n    will span the entire x-dimension.\n    ')
    y_range = Nullable(Instance(Range), help='\n    A range synchronized to the y-dimension of the overlay. If None, the overlay\n    will span the entire y-dimension.\n    ')
    x_interaction = Bool(default=True, help='\n    Whether to respond to horizontal pan motions when an ``x_range`` is present.\n\n    By default, when an ``x_range`` is specified, it is possible to adjust the\n    horizontal position of the range box by panning horizontally inside the\n    box, or along the top or bottom edge of the box. To disable this, and fix\n    the  range box in place horizontally, set to False. (The box will still\n    update if the ``x_range`` is updated programmatically.)\n    ')
    y_interaction = Bool(default=True, help='\n    Whether to respond to vertical pan motions when a ``y_range`` is present.\n\n    By default, when a ``y_range`` is specified, it is possible to adjust the\n    vertical position of the range box by panning vertically inside the box, or\n    along the top or bottom edge of the box. To disable this, and fix the range\n    box in place vertically, set to False. (The box will still update if the\n    ``y_range`` is updated programmatically.)\n    ')
    overlay = Instance(BoxAnnotation, default=DEFAULT_RANGE_OVERLAY, help='\n    A shaded annotation drawn to indicate the configured ranges.\n    ')

    @error(NO_RANGE_TOOL_RANGES)
    def _check_no_range_tool_ranges(self):
        if self.x_range is None and self.y_range is None:
            return 'At least one of RangeTool.x_range or RangeTool.y_range must be configured'