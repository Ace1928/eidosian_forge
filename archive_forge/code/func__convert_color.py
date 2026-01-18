from __future__ import annotations
import io
from typing import TYPE_CHECKING, Any
from bokeh.io import export_png, export_svg, show
from bokeh.io.export import get_screenshot_as_png
from bokeh.layouts import gridplot
from bokeh.models.annotations.labels import Label
from bokeh.palettes import Category10
from bokeh.plotting import figure
import numpy as np
from contourpy import FillType, LineType
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.util.bokeh_util import filled_to_bokeh, lines_to_bokeh
from contourpy.util.renderer import Renderer
def _convert_color(self, color: str) -> str:
    if isinstance(color, str) and color[0] == 'C':
        index = int(color[1:])
        color = self._palette[index]
    return color