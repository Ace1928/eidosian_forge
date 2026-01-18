from __future__ import annotations
from typing import TYPE_CHECKING
import logging # isort:skip
import numpy as np
from ..core.enums import HorizontalLocation, MarkerType, VerticalLocation
from ..core.properties import (
from ..models import (
from ..models.dom import Template
from ..models.tools import (
from ..transform import linear_cmap
from ..util.options import Options
from ._graph import get_graph_kwargs
from ._plot import get_range, get_scale, process_axis_and_grid
from ._stack import double_stack, single_stack
from ._tools import process_active_tools, process_tools_arg
from .contour import ContourRenderer, from_contour
from .glyph_api import _MARKER_SHORTCUTS, GlyphAPI
class BaseFigureOptions(Options):
    tools = Either(String, Seq(Either(String, Instance(Tool))), default=DEFAULT_TOOLS, help='\n    Tools the plot should start with.\n    ')
    x_minor_ticks = Either(Auto, Int, default='auto', help='\n    Number of minor ticks between adjacent x-axis major ticks.\n    ')
    y_minor_ticks = Either(Auto, Int, default='auto', help='\n    Number of minor ticks between adjacent y-axis major ticks.\n    ')
    x_axis_location = Nullable(Enum(VerticalLocation), default='below', help='\n    Where the x-axis should be located.\n    ')
    y_axis_location = Nullable(Enum(HorizontalLocation), default='left', help='\n    Where the y-axis should be located.\n    ')
    x_axis_label = Nullable(TextLike, default='', help='\n    A label for the x-axis.\n    ')
    y_axis_label = Nullable(TextLike, default='', help='\n    A label for the y-axis.\n    ')
    active_drag = Nullable(Either(Auto, String, Instance(Drag)), default='auto', help='\n    Which drag tool should initially be active.\n    ')
    active_inspect = Nullable(Either(Auto, String, Instance(InspectTool), Seq(Instance(InspectTool))), default='auto', help='\n    Which drag tool should initially be active.\n    ')
    active_scroll = Nullable(Either(Auto, String, Instance(Scroll)), default='auto', help='\n    Which scroll tool should initially be active.\n    ')
    active_tap = Nullable(Either(Auto, String, Instance(Tap)), default='auto', help='\n    Which tap tool should initially be active.\n    ')
    active_multi = Nullable(Either(Auto, String, Instance(GestureTool)), default='auto', help='\n    Specify an active multi-gesture tool, for instance an edit tool or a range tool.\n    ')
    tooltips = Nullable(Either(Instance(Template), String, List(Tuple(String, String))), help='\n    An optional argument to configure tooltips for the Figure. This argument\n    accepts the same values as the ``HoverTool.tooltips`` property. If a hover\n    tool is specified in the ``tools`` argument, this value will override that\n    hover tools ``tooltips`` value. If no hover tool is specified in the\n    ``tools`` argument, then passing tooltips here will cause one to be created\n    and added.\n    ')