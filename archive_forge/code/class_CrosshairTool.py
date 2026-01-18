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
class CrosshairTool(InspectTool):
    """ *toolbar icon*: |crosshair_icon|

    The crosshair tool is a passive inspector tool. It is generally on at all
    times, but can be configured in the inspector's menu associated with the
    *toolbar icon* shown above.

    The crosshair tool draws a crosshair annotation over the plot, centered on
    the current mouse position. The crosshair tool may be configured to draw
    across only one dimension by setting the ``dimension`` property to only
    ``width`` or ``height``.

    .. |crosshair_icon| image:: /_images/icons/Crosshair.png
        :height: 24px
        :alt: Icon of circle with aiming reticle marks representing the crosshair tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    overlay = Either(Auto, Instance(Span), Tuple(Instance(Span), Instance(Span)), default='auto', help='\n    An annotation drawn to indicate the crosshair.\n\n    If ``"auto"``, this will create spans depending on the ``dimensions``\n    property, which based on its value, will result in either one span\n    (horizontal or vertical) or two spans (horizontal and vertical).\n\n    Alternatively the user can provide one ``Span`` instance, where the\n    dimension is indicated by the ``dimension`` property of the ``Span``.\n    Also two ``Span`` instances can be provided. Providing explicit\n    ``Span`` instances allows for constructing linked crosshair, when\n    those instances are shared between crosshair tools of different plots.\n\n    .. note::\n        This property is experimental and may change at any point. In\n        particular in future this will allow using other annotations\n        than ``Span`` and annotation groups.\n    ')
    dimensions = Enum(Dimensions, default='both', help='\n    Which dimensions the crosshair tool is to track. By default, both vertical\n    and horizontal lines will be drawn. If only "width" is supplied, only a\n    horizontal line will be drawn. If only "height" is supplied, only a\n    vertical line will be drawn.\n    ')
    line_color = Color(default='black', help='\n    A color to use to stroke paths with.\n    ')
    line_alpha = Alpha(help='\n    An alpha value to use to stroke paths with.\n    ')
    line_width = Float(default=1, help='\n    Stroke width in units of pixels.\n    ')