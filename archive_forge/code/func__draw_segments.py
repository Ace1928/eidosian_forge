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
def _draw_segments(data: pd.DataFrame, ax: Axes, **params: Any):
    """
    Draw independent line segments between all the
    points
    """
    from matplotlib.collections import LineCollection
    color = to_rgba(data['color'], data['alpha'])
    indices: list[int] = []
    _segments = []
    for _, df in data.groupby('group'):
        idx = df.index
        indices.extend(idx[:-1].to_list())
        x = data['x'].iloc[idx]
        y = data['y'].iloc[idx]
        _segments.append(make_line_segments(x, y, ispath=True))
    segments = np.vstack(_segments).tolist()
    edgecolor = color if color is None else [color[i] for i in indices]
    linewidth = data.loc[indices, 'size']
    linestyle = data.loc[indices, 'linetype']
    coll = LineCollection(segments, edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle, capstyle=params.get('lineend', None), zorder=params['zorder'], rasterized=params['raster'])
    ax.add_collection(coll)