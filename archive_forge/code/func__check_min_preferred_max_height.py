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
@error(MIN_PREFERRED_MAX_HEIGHT)
def _check_min_preferred_max_height(self):
    min_height = self.min_height if self.min_height is not None else 0
    height = self.height if self.height is not None and (self.sizing_mode == 'fixed' or self.height_policy == 'fixed') else min_height
    max_height = self.max_height if self.max_height is not None else height
    if not min_height <= height <= max_height:
        return str(self)