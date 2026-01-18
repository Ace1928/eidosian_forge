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
class LassoSelectTool(Drag, RegionSelectTool):
    """ *toolbar icon*: |lasso_select_icon|

    The lasso selection tool allows users to make selections on a Plot by
    indicating a free-drawn "lasso" region by dragging the mouse or a finger
    over the plot region. The end of the drag event indicates the selection
    region is ready.

    See :ref:`ug_styling_plots_selected_unselected_glyphs` for information
    on styling selected and unselected glyphs.

    .. note::
        Selections can be comprised of multiple regions, even those made by
        different selection tools. Hold down the SHIFT key while making a
        selection to append the new selection to any previous selection that
        might exist.

    .. |lasso_select_icon| image:: /_images/icons/LassoSelect.png
        :height: 24px
        :alt:  Icon of a looped lasso shape representing the lasso-selection tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    overlay = Instance(PolyAnnotation, default=DEFAULT_POLY_OVERLAY, help='\n    A shaded annotation drawn to indicate the selection region.\n    ')
    continuous = Override(default=True)