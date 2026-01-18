from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
class breaks_extended:
    """
    An extension of Wilkinson's tick position algorithm

    Parameters
    ----------
    n : int
        Desired number of breaks
    Q : list
        List of nice numbers
    only_inside : bool
        If ``True``, then all the breaks will be within the given
        range.
    w : list
        Weights applied to the four optimization components
        (simplicity, coverage, density, and legibility). They
        should add up to 1.

    Examples
    --------
    >>> limits = (0, 9)
    >>> breaks_extended()(limits)
    array([  0. ,   2.5,   5. ,   7.5,  10. ])
    >>> breaks_extended(n=6)(limits)
    array([  0.,   2.,   4.,   6.,   8.,  10.])

    References
    ----------
    - Talbot, J., Lin, S., Hanrahan, P. (2010) An Extension of
      Wilkinson's Algorithm for Positioning Tick Labels on Axes,
      InfoVis 2010.

    Additional Credit to Justin Talbot on whose code this
    implementation is almost entirely based.
    """

    def __init__(self, n: int=5, Q: Sequence[float]=(1, 5, 2, 2.5, 4, 3), only_inside: bool=False, w: Sequence[float]=(0.25, 0.2, 0.5, 0.05)):
        self.Q = Q
        self.only_inside = only_inside
        self.w = w
        self.n = n
        self.Q_index = {q: i for i, q in enumerate(Q)}

    def coverage(self, dmin: float, dmax: float, lmin: float, lmax: float) -> float:
        p1 = (dmax - lmax) ** 2
        p2 = (dmin - lmin) ** 2
        p3 = (0.1 * (dmax - dmin)) ** 2
        return 1 - 0.5 * (p1 + p2) / p3

    def coverage_max(self, dmin: float, dmax: float, span: float) -> float:
        range = dmax - dmin
        if span > range:
            half = (span - range) / 2.0
            return 1 - half ** 2 / (0.1 * range) ** 2
        else:
            return 1

    def density(self, k: float, dmin: float, dmax: float, lmin: float, lmax: float) -> float:
        r = (k - 1.0) / (lmax - lmin)
        rt = (self.n - 1) / (max(lmax, dmax) - min(lmin, dmin))
        return 2 - max(r / rt, rt / r)

    def density_max(self, k: float) -> float:
        if k >= self.n:
            return 2 - (k - 1.0) / (self.n - 1.0)
        else:
            return 1

    def simplicity(self, q: float, j: float, lmin: float, lmax: float, lstep: float) -> float:
        eps = 1e-10
        n = len(self.Q)
        i = self.Q_index[q] + 1
        if (lmin % lstep < eps or lstep - lmin % lstep < eps) and lmin <= 0 and (lmax >= 0):
            v = 1
        else:
            v = 0
        return (n - i) / (n - 1.0) + v - j

    def simplicity_max(self, q: float, j: float) -> float:
        n = len(self.Q)
        i = self.Q_index[q] + 1
        v = 1
        return (n - i) / (n - 1.0) + v - j

    def legibility(self, lmin: float, lmax: float, lstep: float) -> float:
        return 1

    def __call__(self, limits: TupleFloat2) -> NDArrayFloat:
        """
        Calculate the breaks

        Parameters
        ----------
        limits : array
            Minimum and maximum values.

        Returns
        -------
        out : array_like
            Sequence of break points.
        """
        Q = self.Q
        w = self.w
        only_inside = self.only_inside
        simplicity_max = self.simplicity_max
        density_max = self.density_max
        coverage_max = self.coverage_max
        simplicity = self.simplicity
        coverage = self.coverage
        density = self.density
        legibility = self.legibility
        log10 = np.log10
        ceil = np.ceil
        floor = np.floor
        dmin, dmax = (float(limits[0]), float(limits[1]))
        if dmin > dmax:
            dmin, dmax = (dmax, dmin)
        elif dmin == dmax:
            return np.array([dmin])
        best_score = -2.0
        best: TupleFloat5 = (0, 0, 0, 0, 0)
        j = 1.0
        while j < float('inf'):
            for q in Q:
                sm = simplicity_max(q, j)
                if w[0] * sm + w[1] + w[2] + w[3] < best_score:
                    j = float('inf')
                    break
                k = 2.0
                while k < float('inf'):
                    dm = density_max(k)
                    if w[0] * sm + w[1] + w[2] * dm + w[3] < best_score:
                        break
                    delta = (dmax - dmin) / (k + 1) / j / q
                    z: float = ceil(log10(delta))
                    while z < float('inf'):
                        step = j * q * 10 ** z
                        cm = coverage_max(dmin, dmax, step * (k - 1))
                        if w[0] * sm + w[1] * cm + w[2] * dm + w[3] < best_score:
                            break
                        min_start = int(floor(dmax / step) * j - (k - 1) * j)
                        max_start = int(ceil(dmin / step) * j)
                        if min_start > max_start:
                            z = z + 1
                            break
                        for start in range(min_start, max_start + 1):
                            lmin = start * (step / j)
                            lmax = lmin + step * (k - 1)
                            lstep = step
                            s = simplicity(q, j, lmin, lmax, lstep)
                            c = coverage(dmin, dmax, lmin, lmax)
                            d = density(k, dmin, dmax, lmin, lmax)
                            l = legibility(lmin, lmax, lstep)
                            score = w[0] * s + w[1] * c + w[2] * d + w[3] * l
                            if score > best_score and (not only_inside or (lmin >= dmin and lmax <= dmax)):
                                best_score = score
                                best = (lmin, lmax, lstep, q, k)
                        z = z + 1
                    k = k + 1
            j = j + 1
        locs = best[0] + np.arange(best[4]) * best[2]
        return locs