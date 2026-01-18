from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
class breaks_log:
    """
    Integer breaks on log transformed scales

    Parameters
    ----------
    n : int
        Desired number of breaks
    base : int
        Base of logarithm

    Examples
    --------
    >>> x = np.logspace(3, 6)
    >>> limits = min(x), max(x)
    >>> breaks_log()(limits)
    array([     1000,    10000,   100000,  1000000])
    >>> breaks_log(2)(limits)
    array([  1000, 100000])
    >>> breaks_log()([0.1, 1])
    array([0.1, 0.3, 1. , 3. ])
    """

    def __init__(self, n: int=5, base: float=10):
        self.n = n
        self.base = base

    def __call__(self, limits: TupleFloat2) -> NDArrayFloat:
        """
        Compute breaks

        Parameters
        ----------
        limits : tuple
            Minimum and maximum values

        Returns
        -------
        out : array_like
            Sequence of breaks points
        """
        if any(np.isinf(limits)):
            return np.array([])
        n = self.n
        base = self.base
        rng = log(limits, base)
        _min = int(np.floor(rng[0]))
        _max = int(np.ceil(rng[1]))
        if float(base) ** _max > sys.maxsize:
            base = float(base)
        if _max == _min:
            return np.array([base ** _min])
        by = int(np.floor((_max - _min) / n)) + 1
        for step in range(by, 0, -1):
            breaks = np.array([base ** i for i in range(_min, _max + 1, step)])
            relevant_breaks = (limits[0] <= breaks) & (breaks <= limits[1])
            if np.sum(relevant_breaks) >= n - 2:
                return breaks
        return _breaks_log_sub(n=n, base=base)(limits)