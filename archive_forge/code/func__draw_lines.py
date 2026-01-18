from __future__ import annotations
import typing
from collections import Counter
from contextlib import suppress
from warnings import warn
import numpy as np
from .._utils import SIZE_FACTOR, make_line_segments, match, to_rgba
from ..doctools import document
from ..exceptions import PlotnineWarning
from .geom import geom
def _draw_lines(data: pd.DataFrame, ax: Axes, **params: Any):
    """
    Draw a path with the same characteristics from the
    first point to the last point
    """
    from matplotlib.lines import Line2D
    color = to_rgba(data['color'].iloc[0], data['alpha'].iloc[0])
    join_style = _get_joinstyle(data, params)
    lines = Line2D(data['x'], data['y'], color=color, linewidth=data['size'].iloc[0], linestyle=data['linetype'].iloc[0], zorder=params['zorder'], rasterized=params['raster'], **join_style)
    ax.add_artist(lines)