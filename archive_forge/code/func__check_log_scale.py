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
def _check_log_scale(base: Optional[float], sides: str, panel_params: panel_view, coord: coord) -> TupleFloat2:
    """
        Check the log transforms

        Parameters
        ----------
        base : float | None
            Base of the logarithm in which the ticks will be
            calculated. If `None`, the base of the log transform
            the scale will be used.
        sides : str, default="bl"
            Sides onto which to draw the marks. Any combination
            chosen from the characters `btlr`, for *bottom*, *top*,
            *left* or *right* side marks. If `coord_flip()` is used,
            these are the sides *before* the flip.
        panel_params : panel_view
            `x` and `y` view scale values.
        coord : coord
            Coordinate (e.g. coord_cartesian) system of the geom.

        Returns
        -------
        out : tuple
            The bases (base_x, base_y) to use when generating the ticks.
        """

    def is_log_trans(t: trans) -> bool:
        return hasattr(t, 'base') and t.__class__.__name__.startswith('log')

    def get_base(sc, ubase: Optional[float]) -> float:
        ae = sc.aesthetics[0]
        if not isinstance(sc, ScaleContinuous) or not is_log_trans(sc.trans):
            warnings.warn(f'annotation_logticks for {ae}-axis which does not have a log scale. The logticks may not make sense.', PlotnineWarning)
            return 10 if ubase is None else ubase
        base = sc.trans.base
        if ubase is not None and base != ubase:
            warnings.warn(f'The x-axis is log transformed in base={base} ,but the annotation_logticks are computed in base={ubase}', PlotnineWarning)
            return ubase
        return base
    base_x, base_y = (10, 10)
    x_scale = panel_params.x.scale
    y_scale = panel_params.y.scale
    if isinstance(coord, coord_flip):
        x_scale, y_scale = (y_scale, x_scale)
        base_x, base_y = (base_y, base_x)
    if 't' in sides or 'b' in sides:
        base_x = get_base(x_scale, base)
    if 'l' in sides or 'r' in sides:
        base_y = get_base(y_scale, base)
    return (base_x, base_y)