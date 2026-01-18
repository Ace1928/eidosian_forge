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
@staticmethod
def _calc_ticks(value_range: TupleFloat2, base: float) -> tuple[AnyArray, AnyArray, AnyArray]:
    """
        Calculate tick marks within a range

        Parameters
        ----------
        value_range: tuple
            Range for which to calculate ticks.

        base : number
            Base of logarithm

        Returns
        -------
        out: tuple
            (major, middle, minor) tick locations
        """

    def _minor(x: Sequence[Any], mid_idx: int) -> AnyArray:
        return np.hstack([x[1:mid_idx], x[mid_idx + 1:-1]])
    low = np.floor(value_range[0])
    high = np.ceil(value_range[1])
    arr = base ** np.arange(low, float(high + 1))
    n_ticks = int(np.round(base) - 1)
    breaks = [log(np.linspace(b1, b2, n_ticks + 1), base) for b1, b2 in list(zip(arr, arr[1:]))]
    major = np.array([x[0] for x in breaks] + [breaks[-1][-1]])
    if n_ticks % 2:
        mid_idx = n_ticks // 2
        middle = np.array([x[mid_idx] for x in breaks])
        minor = np.hstack([_minor(x, mid_idx) for x in breaks])
    else:
        middle = np.array([])
        minor = np.hstack([x[1:-1] for x in breaks])
    return (major, middle, minor)