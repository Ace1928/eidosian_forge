from __future__ import annotations
from collections.abc import Sequence
import io
from typing import TYPE_CHECKING, Any, cast
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
from contourpy import FillType, LineType
from contourpy.convert import convert_filled, convert_lines
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.util.mpl_util import filled_to_mpl_paths, lines_to_mpl_paths
from contourpy.util.renderer import Renderer
def _arrow(self, ax: Axes, line_start: cpy.CoordinateArray, line_end: cpy.CoordinateArray, color: str, alpha: float, arrow_size: float) -> None:
    mid = 0.5 * (line_start + line_end)
    along = line_end - line_start
    along /= np.sqrt(np.dot(along, along))
    right = np.asarray((along[1], -along[0]))
    arrow = np.stack((mid - (along * 0.5 - right) * arrow_size, mid + along * 0.5 * arrow_size, mid - (along * 0.5 + right) * arrow_size))
    ax.plot(arrow[:, 0], arrow[:, 1], '-', c=color, alpha=alpha)