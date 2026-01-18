from __future__ import annotations
import logging # isort:skip
from ..colors import RGB, Color, ColorLike
from ..core.enums import (
from ..core.has_props import HasProps, abstract
from ..core.properties import (
from ..core.property.struct import Optional
from ..core.property_aliases import GridSpacing, Pixels, TracksSizing
from ..core.validation import error, warning
from ..core.validation.errors import (
from ..core.validation.warnings import (
from ..model import Model
from .ui.panes import Pane
from .ui.tooltips import Tooltip
from .ui.ui_element import UIElement
@abstract
class GridCommon(HasProps):
    """ Common properties for grid-like layouts. """
    rows = Nullable(TracksSizing, default=None, help="\n    Describes how the grid should maintain its rows' heights.\n\n    This maps to CSS grid's track sizing options. In particular the following\n    values are allowed:\n\n    * length, e.g. ``100px``, ``5.5em``\n    * percentage, e.g. ``33%``\n    * flex, e.g. 1fr\n    * enums, e.g.  ``max-content``, ``min-content``, ``auto``, etc.\n\n    If a single value is provided, then it applies to all rows. A list of\n    values can be provided to size all rows, or a dictionary providing\n    sizing for individual rows.\n\n    See https://developer.mozilla.org/en-US/docs/Web/CSS/grid-template-rows\n    or https://w3c.github.io/csswg-drafts/css-grid/#track-sizing for details.\n    ")
    cols = Nullable(TracksSizing, default=None, help="\n    Describes how the grid should maintain its columns' widths.\n\n    This maps to CSS grid's track sizing options. In particular the following\n    values are allowed:\n\n    * length, e.g. ``100px``, ``5.5em``\n    * percentage, e.g. ``33%``\n    * flex, e.g. 1fr\n    * enums, e.g.  ``max-content``, ``min-content``, ``auto``, etc.\n\n    If a single value is provided, then it applies to all columns. A list of\n    values can be provided to size all columns, or a dictionary providing\n    sizing for individual columns.\n\n    See https://developer.mozilla.org/en-US/docs/Web/CSS/grid-template-columns\n    or https://w3c.github.io/csswg-drafts/css-grid/#track-sizing for details.\n    ")
    spacing = GridSpacing(default=0, help='\n    The gap between children (in pixels).\n\n    Either a number, if spacing is the same for both dimensions, or a pair\n    of numbers indicating spacing in the vertical and horizontal dimensions\n    respectively.\n    ')