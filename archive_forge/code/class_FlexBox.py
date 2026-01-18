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
class FlexBox(LayoutDOM):
    """ Abstract base class for Row and Column. Do not use directly.

    """

    def __init__(self, *args, **kwargs) -> None:
        if len(args) > 0 and 'children' in kwargs:
            raise ValueError("'children' keyword cannot be used with positional arguments")
        elif len(args) > 0:
            kwargs['children'] = list(args)
        super().__init__(**kwargs)

    @warning(EMPTY_LAYOUT)
    def _check_empty_layout(self):
        from itertools import chain
        if not list(chain(self.children)):
            return str(self)

    @warning(BOTH_CHILD_AND_ROOT)
    def _check_child_is_also_root(self):
        problems = []
        for c in self.children:
            if c.document is not None and c in c.document.roots:
                problems.append(str(c))
        if problems:
            return ', '.join(problems)
        else:
            return None

    @error(REPEATED_LAYOUT_CHILD)
    def _check_repeated_layout_children(self):
        if len(self.children) != len(set(self.children)):
            return str(self)
    children = List(Instance(UIElement), help='\n    The list of children, which can be other components including plots, rows, columns, and widgets.\n    ')
    spacing = Int(default=0, help='\n    The gap between children (in pixels).\n    ')