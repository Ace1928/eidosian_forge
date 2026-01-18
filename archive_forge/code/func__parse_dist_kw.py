from __future__ import annotations
import math
import warnings
from collections import namedtuple
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py
from ._fit import FitResult
from ._stats_py import find_repeats, _normtest_finish, SignificanceResult
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr
from ._axis_nan_policy import _axis_nan_policy_factory
def _parse_dist_kw(dist, enforce_subclass=True):
    """Parse `dist` keyword.

    Parameters
    ----------
    dist : str or stats.distributions instance.
        Several functions take `dist` as a keyword, hence this utility
        function.
    enforce_subclass : bool, optional
        If True (default), `dist` needs to be a
        `_distn_infrastructure.rv_generic` instance.
        It can sometimes be useful to set this keyword to False, if a function
        wants to accept objects that just look somewhat like such an instance
        (for example, they have a ``ppf`` method).

    """
    if isinstance(dist, rv_generic):
        pass
    elif isinstance(dist, str):
        try:
            dist = getattr(distributions, dist)
        except AttributeError as e:
            raise ValueError('%s is not a valid distribution name' % dist) from e
    elif enforce_subclass:
        msg = '`dist` should be a stats.distributions instance or a string with the name of such a distribution.'
        raise ValueError(msg)
    return dist