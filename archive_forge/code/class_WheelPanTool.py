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
class WheelPanTool(Scroll):
    """ *toolbar icon*: |wheel_pan_icon|

    The wheel pan tool allows the user to pan the plot along the configured
    dimension using the scroll wheel.

    .. |wheel_pan_icon| image:: /_images/icons/WheelPan.png
        :height: 24px
        :alt: Icon of a mouse shape next to crossed arrows representing the wheel-pan tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    dimension = Enum(Dimension, default='width', help='\n    Which dimension the wheel pan tool is constrained to act in. By default the\n    wheel pan tool will pan the plot along the x-axis.\n    ')