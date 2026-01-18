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
def _axes_get_size_inches(ax: Axes) -> TupleFloat2:
    """
    Size of axes in inches

    Parameters
    ----------
    ax : axes
        Axes

    Returns
    -------
    out : tuple[float, float]
        (width, height) of ax in inches
    """
    fig = ax.get_figure()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return (bbox.width, bbox.height)