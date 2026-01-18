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