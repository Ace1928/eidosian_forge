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
class FigureOptions(BaseFigureOptions):
    x_range = RangeLike(default=InstanceDefault(DataRange1d), help='\n    Customize the x-range of the plot.\n    ')
    y_range = RangeLike(default=InstanceDefault(DataRange1d), help='\n    Customize the y-range of the plot.\n    ')
    x_axis_type = AxisType(default='auto', help='\n    The type of the x-axis.\n    ')
    y_axis_type = AxisType(default='auto', help='\n    The type of the y-axis.\n    ')