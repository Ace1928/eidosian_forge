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
class GroupBox(LayoutDOM):
    """ A panel that allows to group UI elements.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    title = Nullable(String, help='\n    The title text of the group. If not provided, only the frame will be showed.\n    ')
    child = Instance(UIElement, help='\n    The child UI element. This can be a single UI control, widget, etc., or\n    a container layout like ``Column`` or ``Row``, or a combitation of layouts.\n    ')
    checkable = Bool(False, help='\n    Whether to allow disabling this group (all its children) via a checkbox\n    in the UI. This allows to broadcast ``disabled`` state across multiple\n    UI controls that support that state.\n    ')