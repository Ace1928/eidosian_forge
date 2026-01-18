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
class PointDrawTool(EditTool, Drag, Tap):
    """ *toolbar icon*: |point_draw_icon|

    The PointDrawTool allows adding, dragging and deleting point-like glyphs
    (i.e subclasses of ``XYGlyph``) on one or more renderers by editing the
    underlying ``ColumnDataSource`` data. Like other drawing tools, the
    renderers that are to be edited must be supplied explicitly as a list. Any
    newly added points will be inserted on the ``ColumnDataSource`` of the
    first supplied renderer.

    The tool will modify the columns on the data source corresponding to the
    ``x`` and ``y`` values of the glyph. Any additional columns in the data
    source will be padded with the given ``empty_value`` when adding a new
    point.

    .. note::
        The data source updates will trigger data change events continuously
        throughout the edit operations on the BokehJS side. In Bokeh server
        apps, the data source will only be synced once, when the edit operation
        finishes.

    The supported actions include:

    * Add point: Tap anywhere on the plot

    * Move point: Tap and drag an existing point, the point will be
      dropped once you let go of the mouse button.

    * Delete point: Tap a point to select it then press BACKSPACE
      key while the mouse is within the plot area.

    .. |point_draw_icon| image:: /_images/icons/PointDraw.png
        :height: 24px
        :alt: Icon of three points with an arrow pointing to one representing the point-edit tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    renderers = List(GlyphRendererOf(XYGlyph), help='\n    A list of renderers corresponding to glyphs that may be edited.\n    ')
    add = Bool(default=True, help='\n    Enables adding of new points on tap events.\n    ')
    drag = Bool(default=True, help='\n    Enables dragging of existing points on pan events.\n    ')
    num_objects = Int(default=0, help='\n    Defines a limit on the number of points that can be drawn. By default there\n    is no limit on the number of objects, but if enabled the oldest drawn point\n    will be dropped to make space for the new point.\n    ')