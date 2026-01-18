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
class Tabs(LayoutDOM):
    """ A panel widget with navigation tabs.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/interaction/widgets/tab_panes.py'
    tabs = List(Instance(TabPanel), help='\n    The list of child panel widgets.\n    ').accepts(List(Tuple(String, Instance(UIElement))), lambda items: [TabPanel(title=title, child=child) for title, child in items])
    tabs_location = Enum(Location, default='above', help='\n    The location of the buttons that activate tabs.\n    ')
    active = Int(0, help='\n    The index of the active tab.\n    ')