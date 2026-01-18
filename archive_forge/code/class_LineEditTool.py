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
class LineEditTool(EditTool, Drag, Tap):
    """ *toolbar icon*: |line_edit_icon|

    The LineEditTool allows editing the intersection points of one or more ``Line`` glyphs.
    Glyphs to be edited are defined via the ``renderers``
    property and a renderer for the intersections is set via the ``intersection_renderer``
    property (must render a point-like Glyph (a subclass of ``XYGlyph``).

    The tool will modify the columns on the data source corresponding to the
    ``x`` and ``y`` values of the glyph. Any additional columns in the data
    source will be padded with the declared ``empty_value``, when adding a new
    point.

    The supported actions include:

    * Show intersections: press an existing line

    * Move point: Drag an existing point and let go of the mouse button to
      release it.

    .. |line_edit_icon| image:: /_images/icons/LineEdit.png
        :height: 24px
        :alt: Icon of a line with a point on it with an arrow pointing at it representing the line-edit tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    renderers = List(GlyphRendererOf(Line), help='\n    A list of renderers corresponding to glyphs that may be edited.\n    ')
    intersection_renderer = GlyphRendererOf(LineGlyph)(help='\n    The renderer used to render the intersections of a selected line\n    ')
    dimensions = Enum(Dimensions, default='both', help='\n    Which dimensions this edit tool is constrained to act in. By default\n    the line edit tool allows moving points in any dimension, but can be\n    configured to only allow horizontal movement across the width of the\n    plot, or vertical across the height of the plot.\n    ')