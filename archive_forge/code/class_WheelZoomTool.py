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
class WheelZoomTool(Scroll):
    """ *toolbar icon*: |wheel_zoom_icon|

    The wheel zoom tool will zoom the plot in and out, centered on the
    current mouse location.

    The wheel zoom tool also activates the border regions of a Plot for
    "single axis" zooming. For instance, zooming in the vertical border or
    axis will effect a zoom in the vertical direction only, with the
    horizontal dimension kept fixed.

    .. |wheel_zoom_icon| image:: /_images/icons/WheelZoom.png
        :height: 24px
        :alt: Icon of a mouse shape next to an hourglass representing the wheel-zoom tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    dimensions = Enum(Dimensions, default='both', help='\n    Which dimensions the wheel zoom tool is constrained to act in. By default\n    the wheel zoom tool will zoom in any dimension, but can be configured to\n    only zoom horizontally across the width of the plot, or vertically across\n    the height of the plot.\n    ')
    renderers = Either(Auto, List(Instance(DataRenderer)), default='auto', help='\n    Restrict zoom to ranges used by the provided data renderers. If ``"auto"``\n    then all ranges provided by the cartesian frame will be used.\n    ')
    level = NonNegative(Int, default=0, help='\n    When working with composite scales (sub-coordinates), this property\n    allows to configure which set of ranges to scale. The default is to\n    scale top-level (frame) ranges.\n    ')
    maintain_focus = Bool(default=True, help='\n    If True, then hitting a range bound in any one dimension will prevent all\n    further zooming all dimensions. If False, zooming can continue\n    independently in any dimension that has not yet reached its bounds, even if\n    that causes overall focus or aspect ratio to change.\n    ')
    zoom_on_axis = Bool(default=True, help='\n    Whether scrolling on an axis (outside the central plot area) should zoom\n    that dimension. If enabled, the behavior of this feature can be configured\n    with ``zoom_together`` property.\n    ')
    zoom_together = Enum('none', 'cross', 'all', default='all', help='\n    Defines the behavior of the tool when zooming on an axis:\n\n    - ``"none"``\n        zoom only the axis that\'s being interacted with. Any cross\n        axes nor any other axes in the dimension of this axis will be affected.\n    - ``"cross"``\n        zoom the axis that\'s being interacted with and its cross\n        axis, if configured. No other axes in this or cross dimension will be\n        affected.\n    - ``"all"``\n        zoom all axes in the dimension of the axis that\'s being\n        interacted with. All cross axes will be unaffected.\n    ')
    speed = Float(default=1 / 600, help='\n    Speed at which the wheel zooms. Default is 1/600. Optimal range is between\n    0.001 and 0.09. High values will be clipped. Speed may very between browsers.\n    ')