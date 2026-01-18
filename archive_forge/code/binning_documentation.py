from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from ..exceptions import PlotnineError
from ..scales.scale_discrete import scale_discrete

    Compute fuzzy breaks

    For a continuous scale, fuzzybreaks "preserve" the range of
    the scale. The fuzzing is close to numerical roundoff and
    is visually imperceptible.

    Parameters
    ----------
    scale : scale
        Scale
    breaks : array_like
        Sequence of break points. If provided and the scale is not
        discrete, they are returned.
    boundary : float
        First break. If `None` a suitable on is computed using
        the range of the scale and the binwidth.
    binwidth : float
        Separation between the breaks
    bins : int
        Number of bins
    right : bool
        If `True` the right edges of the bins are part of the
        bin. If `False` then the left edges of the bins are part
        of the bin.

    Returns
    -------
    out : array_like
    