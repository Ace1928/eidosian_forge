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
class ScrollBox(LayoutDOM):
    """ A panel that allows to scroll overflowing UI elements.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    child = Instance(UIElement, help='\n    The child UI element. This can be a single UI control, widget, etc., or\n    a container layout like ``Column`` or ``Row``, or a combitation of layouts.\n    ')
    horizontal_scrollbar = Enum(ScrollbarPolicy, default='auto', help='\n    The visibility of the horizontal scrollbar.\n    ')
    vertical_scrollbar = Enum(ScrollbarPolicy, default='auto', help='\n    The visibility of the vertical scrollbar.\n    ')