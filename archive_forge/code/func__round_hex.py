from __future__ import annotations
import logging # isort:skip
from typing import Any
import numpy as np
from .dependencies import import_required
def _round_hex(q: Any, r: Any) -> tuple[Any, Any]:
    """ Round floating point axial hex coordinates to integer *(q,r)*
    coordinates.

    This code was adapted from:

        https://www.redblobgames.com/grids/hexagons/#rounding

    Args:
        q (array[float]) :
            NumPy array of Floating point axial *q* coordinates to round

        r (array[float]) :
            NumPy array of Floating point axial *q* coordinates to round

    Returns:
        (array[int], array[int])

    """
    x = q
    z = r
    y = -x - z
    rx = np.round(x)
    ry = np.round(y)
    rz = np.round(z)
    dx = np.abs(rx - x)
    dy = np.abs(ry - y)
    dz = np.abs(rz - z)
    cond = (dx > dy) & (dx > dz)
    q = np.where(cond, -(ry + rz), rx)
    r = np.where(~cond & ~(dy > dz), -(rx + ry), rz)
    return (q.astype(int), r.astype(int))