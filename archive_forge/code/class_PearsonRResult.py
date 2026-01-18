import warnings
import math
from math import gcd
from collections import namedtuple
import numpy as np
from numpy import array, asarray, ma
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements
from scipy._lib._util import (check_random_state, MapWrapper, _get_nan,
import scipy.special as special
from scipy import linalg
from . import distributions
from . import _mstats_basic as mstats_basic
from ._stats_mstats_common import (_find_repeats, linregress, theilslopes,
from ._stats import (_kendall_dis, _toint64, _weightedrankedtau,
from dataclasses import dataclass, field
from ._hypotests import _all_partitions
from ._stats_pythran import _compute_outer_prob_inside_method
from ._resampling import (MonteCarloMethod, PermutationMethod, BootstrapMethod,
from ._axis_nan_policy import (_axis_nan_policy_factory,
from ._binomtest import _binary_search_for_binom_tst as _binary_search
from scipy._lib._bunch import _make_tuple_bunch
from scipy import stats
from scipy.optimize import root_scalar
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy._lib._util import normalize_axis_index
from scipy._lib._util import float_factorial  # noqa: F401
from scipy.stats._mstats_basic import (  # noqa: F401
class PearsonRResult(PearsonRResultBase):
    """
    Result of `scipy.stats.pearsonr`

    Attributes
    ----------
    statistic : float
        Pearson product-moment correlation coefficient.
    pvalue : float
        The p-value associated with the chosen alternative.

    Methods
    -------
    confidence_interval
        Computes the confidence interval of the correlation
        coefficient `statistic` for the given confidence level.

    """

    def __init__(self, statistic, pvalue, alternative, n, x, y):
        super().__init__(statistic, pvalue)
        self._alternative = alternative
        self._n = n
        self._x = x
        self._y = y
        self.correlation = statistic

    def confidence_interval(self, confidence_level=0.95, method=None):
        """
        The confidence interval for the correlation coefficient.

        Compute the confidence interval for the correlation coefficient
        ``statistic`` with the given confidence level.

        If `method` is not provided,
        The confidence interval is computed using the Fisher transformation
        F(r) = arctanh(r) [1]_.  When the sample pairs are drawn from a
        bivariate normal distribution, F(r) approximately follows a normal
        distribution with standard error ``1/sqrt(n - 3)``, where ``n`` is the
        length of the original samples along the calculation axis. When
        ``n <= 3``, this approximation does not yield a finite, real standard
        error, so we define the confidence interval to be -1 to 1.

        If `method` is an instance of `BootstrapMethod`, the confidence
        interval is computed using `scipy.stats.bootstrap` with the provided
        configuration options and other appropriate settings. In some cases,
        confidence limits may be NaN due to a degenerate resample, and this is
        typical for very small samples (~6 observations).

        Parameters
        ----------
        confidence_level : float
            The confidence level for the calculation of the correlation
            coefficient confidence interval. Default is 0.95.

        method : BootstrapMethod, optional
            Defines the method used to compute the confidence interval. See
            method description for details.

            .. versionadded:: 1.11.0

        Returns
        -------
        ci : namedtuple
            The confidence interval is returned in a ``namedtuple`` with
            fields `low` and `high`.

        References
        ----------
        .. [1] "Pearson correlation coefficient", Wikipedia,
               https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        """
        if isinstance(method, BootstrapMethod):
            ci = _pearsonr_bootstrap_ci(confidence_level, method, self._x, self._y, self._alternative)
        elif method is None:
            ci = _pearsonr_fisher_ci(self.statistic, self._n, confidence_level, self._alternative)
        else:
            message = '`method` must be an instance of `BootstrapMethod` or None.'
            raise ValueError(message)
        return ci