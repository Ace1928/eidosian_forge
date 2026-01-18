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
class PolySelectTool(Tap, RegionSelectTool):
    """ *toolbar icon*: |poly_select_icon|

    The polygon selection tool allows users to make selections on a
    Plot by indicating a polygonal region with mouse clicks. single
    clicks (or taps) add successive points to the definition of the
    polygon, and a press click (or tap) indicates the selection
    region is ready.

    See :ref:`ug_styling_plots_selected_unselected_glyphs` for information
    on styling selected and unselected glyphs.

    .. note::
        Selections can be comprised of multiple regions, even those
        made by different selection tools. Hold down the SHIFT key
        while making a selection to append the new selection to any
        previous selection that might exist.

    .. |poly_select_icon| image:: /_images/icons/PolygonSelect.png
        :height: 24px
        :alt: Icon of a dashed trapezoid with an arrow pointing at the lower right representing the polygon-selection tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    overlay = Instance(PolyAnnotation, default=DEFAULT_POLY_OVERLAY, help='\n    A shaded annotation drawn to indicate the selection region.\n    ')