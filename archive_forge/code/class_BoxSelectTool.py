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
class BoxSelectTool(Drag, RegionSelectTool):
    """ *toolbar icon*: |box_select_icon|

    The box selection tool allows users to make selections on a Plot by showing
    a rectangular region by dragging the mouse or a finger over the plot area.
    The end of the drag event indicates the selection region is ready.

    See :ref:`ug_styling_plots_selected_unselected_glyphs` for information
    on styling selected and unselected glyphs.


    .. |box_select_icon| image:: /_images/icons/BoxSelect.png
        :height: 24px
        :alt: Icon of a dashed box with a + in the lower right representing the box-selection tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    dimensions = Enum(Dimensions, default='both', help='\n    Which dimensions the box selection is to be free in. By default, users may\n    freely draw selections boxes with any dimensions. If only "width" is set,\n    the box will be constrained to span the entire vertical space of the plot,\n    only the horizontal dimension can be controlled. If only "height" is set,\n    the box will be constrained to span the entire horizontal space of the\n    plot, and the vertical dimension can be controlled.\n    ')
    overlay = Instance(BoxAnnotation, default=DEFAULT_BOX_SELECT_OVERLAY, help='\n    A shaded annotation drawn to indicate the selection region.\n    ')
    origin = Enum('corner', 'center', default='corner', help='\n    Indicates whether the rectangular selection area should originate from a corner\n    (top-left or bottom-right depending on direction) or the center of the box.\n    ')