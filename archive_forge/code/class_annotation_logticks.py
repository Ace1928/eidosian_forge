from __future__ import annotations
import typing
import warnings
import numpy as np
import pandas as pd
from .._utils import log
from ..coords import coord_flip
from ..exceptions import PlotnineWarning
from ..scales.scale_continuous import scale_continuous as ScaleContinuous
from .annotate import annotate
from .geom_path import geom_path
from .geom_rug import geom_rug
class annotation_logticks(annotate):
    """
    Marginal log ticks.

    If added to a plot that does not have a log10 axis
    on the respective side, a warning will be issued.

    Parameters
    ----------
    sides :
        Sides onto which to draw the marks. Any combination
        chosen from the characters `btlr`, for *bottom*, *top*,
        *left* or *right* side marks. If `coord_flip()` is used,
        these are the sides *after* the flip.
    alpha :
        Transparency of the ticks
    color :
        Colour of the ticks
    size :
        Thickness of the ticks
    linetype :
        Type of line
    lengths:
        length of the ticks drawn for full / half / tenth
        ticks relative to panel size
    base :
        Base of the logarithm in which the ticks will be
        calculated. If `None`, the base used to log transform
        the scale will be used.
    """

    def __init__(self, sides: str='bl', alpha: float=1, color: str | TupleFloat3 | TupleFloat4='black', size: float=0.5, linetype: Literal['solid', 'dashed', 'dashdot', 'dotted'] | Sequence[float]='solid', lengths: TupleFloat3=(0.036, 0.0225, 0.012), base: float | None=None):
        if len(lengths) != 3:
            raise ValueError('length for annotation_logticks must be a tuple of 3 floats')
        self._annotation_geom = _geom_logticks(sides=sides, alpha=alpha, color=color, size=size, linetype=linetype, lengths=lengths, base=base)