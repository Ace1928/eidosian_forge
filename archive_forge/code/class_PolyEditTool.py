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
class PolyEditTool(PolyTool, Drag, Tap):
    """ *toolbar icon*: |poly_edit_icon|

    The PolyEditTool allows editing the vertices of one or more ``Patches`` or
    ``MultiLine`` glyphs. Glyphs to be edited are defined via the ``renderers``
    property and a renderer for the vertices is set via the ``vertex_renderer``
    property (must render a point-like Glyph (a subclass of ``XYGlyph``).

    The tool will modify the columns on the data source corresponding to the
    ``xs`` and ``ys`` values of the glyph. Any additional columns in the data
    source will be padded with the declared ``empty_value``, when adding a new
    point.

    The supported actions include:

    * Show vertices: press an existing patch or multi-line

    * Add vertex: press an existing vertex to select it, the tool will
      draw the next point, to add it tap in a new location. To finish editing
      and add a point press otherwise press the ESC key to cancel.

    * Move vertex: Drag an existing vertex and let go of the mouse button to
      release it.

    * Delete vertex: After selecting one or more vertices press BACKSPACE
      while the mouse cursor is within the plot area.

    .. |poly_edit_icon| image:: /_images/icons/PolyEdit.png
        :height: 24px
        :alt: Icon of two lines meeting in a vertex with an arrow pointing at it representing the polygon-edit tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    renderers = List(GlyphRendererOf(MultiLine, Patches), help='\n    A list of renderers corresponding to glyphs that may be edited.\n    ')