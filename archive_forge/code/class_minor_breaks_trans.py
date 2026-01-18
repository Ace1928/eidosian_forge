from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
class minor_breaks_trans:
    """
    Compute minor breaks for transformed scales

    The minor breaks are computed in data space.
    This together with major breaks computed in
    transform space reveals the non linearity of
    of a scale. See the log transforms created
    with :func:`log_trans` like :class:`log10_trans`.

    Parameters
    ----------
    trans : trans or type
        Trans object or trans class.
    n : int
        Number of minor breaks between the major
        breaks.

    Examples
    --------
    >>> from mizani.transforms import sqrt_trans
    >>> major = [1, 2, 3, 4]
    >>> limits = [0, 5]
    >>> t1 = sqrt_trans()
    >>> t1.minor_breaks(major, limits)
    array([1.58113883, 2.54950976, 3.53553391])

    # Changing the regular `minor_breaks` method

    >>> t2 = sqrt_trans()
    >>> t2.minor_breaks = minor_breaks()
    >>> t2.minor_breaks(major, limits)
    array([0.5, 1.5, 2.5, 3.5, 4.5])

    More than 1 minor break

    >>> major = [1, 10]
    >>> limits = [1, 10]
    >>> t2.minor_breaks(major, limits, 4)
    array([2.8, 4.6, 6.4, 8.2])
    """

    def __init__(self, trans: Trans, n: int=1):
        self.trans = trans
        self.n = n

    def __call__(self, major: FloatArrayLike, limits: Optional[TupleFloat2]=None, n: Optional[int]=None) -> NDArrayFloat:
        """
        Minor breaks for transformed scales

        Parameters
        ----------
        major : array_like
            Major breaks
        limits : array_like | None
            Limits of the scale. If *array_like*, must be
            of size 2. If **None**, then the minimum and
            maximum of the major breaks are used.
        n : int
            Number of minor breaks between the major
            breaks. If **None**, then *self.n* is used.

        Returns
        -------
        out : array_like
            Minor breaks
        """
        if limits is None:
            limits = min_max(major)
        if n is None:
            n = self.n
        major = self._extend_breaks(major)
        major = self.trans.inverse(major)
        limits = self.trans.inverse(limits)
        minor = minor_breaks(n)(major, limits)
        return self.trans.transform(minor)

    def _extend_breaks(self, major: FloatArrayLike) -> FloatArrayLike:
        """
        Append 2 extra breaks at either end of major

        If breaks of transform space are non-equidistant,
        :func:`minor_breaks` add minor breaks beyond the first
        and last major breaks. The solutions is to extend those
        breaks (in transformed space) before the minor break call
        is made. How the breaks depends on the type of transform.
        """
        trans = self.trans
        trans = trans if isinstance(trans, type) else trans.__class__
        is_log = trans.__name__.startswith('log')
        diff = np.diff(major)
        step = diff[0]
        if is_log and all(diff == step):
            major = np.hstack([major[0] - step, major, major[-1] + step])
        return major