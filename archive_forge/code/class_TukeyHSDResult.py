from collections import namedtuple
from dataclasses import dataclass
from math import comb
import numpy as np
import warnings
from itertools import combinations
import scipy.stats
from scipy.optimize import shgo
from . import distributions
from ._common import ConfidenceInterval
from ._continuous_distns import chi2, norm
from scipy.special import gamma, kv, gammaln
from scipy.fft import ifft
from ._stats_pythran import _a_ij_Aij_Dij2
from ._stats_pythran import (
from ._axis_nan_policy import _axis_nan_policy_factory
from scipy.stats import _stats_py
class TukeyHSDResult:
    """Result of `scipy.stats.tukey_hsd`.

    Attributes
    ----------
    statistic : float ndarray
        The computed statistic of the test for each comparison. The element
        at index ``(i, j)`` is the statistic for the comparison between groups
        ``i`` and ``j``.
    pvalue : float ndarray
        The associated p-value from the studentized range distribution. The
        element at index ``(i, j)`` is the p-value for the comparison
        between groups ``i`` and ``j``.

    Notes
    -----
    The string representation of this object displays the most recently
    calculated confidence interval, and if none have been previously
    calculated, it will evaluate ``confidence_interval()``.

    References
    ----------
    .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "7.4.7.1. Tukey's
           Method."
           https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm,
           28 November 2020.
    """

    def __init__(self, statistic, pvalue, _nobs, _ntreatments, _stand_err):
        self.statistic = statistic
        self.pvalue = pvalue
        self._ntreatments = _ntreatments
        self._nobs = _nobs
        self._stand_err = _stand_err
        self._ci = None
        self._ci_cl = None

    def __str__(self):
        if self._ci is None:
            self.confidence_interval(confidence_level=0.95)
        s = f"Tukey's HSD Pairwise Group Comparisons ({self._ci_cl * 100:.1f}% Confidence Interval)\n"
        s += 'Comparison  Statistic  p-value  Lower CI  Upper CI\n'
        for i in range(self.pvalue.shape[0]):
            for j in range(self.pvalue.shape[0]):
                if i != j:
                    s += f' ({i} - {j}) {self.statistic[i, j]:>10.3f}{self.pvalue[i, j]:>10.3f}{self._ci.low[i, j]:>10.3f}{self._ci.high[i, j]:>10.3f}\n'
        return s

    def confidence_interval(self, confidence_level=0.95):
        """Compute the confidence interval for the specified confidence level.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval
            of the estimated proportion. Default is .95.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence intervals for each
            comparison. The high and low values are accessible for each
            comparison at index ``(i, j)`` between groups ``i`` and ``j``.

        References
        ----------
        .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "7.4.7.1.
               Tukey's Method."
               https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm,
               28 November 2020.

        Examples
        --------
        >>> from scipy.stats import tukey_hsd
        >>> group0 = [24.5, 23.5, 26.4, 27.1, 29.9]
        >>> group1 = [28.4, 34.2, 29.5, 32.2, 30.1]
        >>> group2 = [26.1, 28.3, 24.3, 26.2, 27.8]
        >>> result = tukey_hsd(group0, group1, group2)
        >>> ci = result.confidence_interval()
        >>> ci.low
        array([[-3.649159, -8.249159, -3.909159],
               [ 0.950841, -3.649159,  0.690841],
               [-3.389159, -7.989159, -3.649159]])
        >>> ci.high
        array([[ 3.649159, -0.950841,  3.389159],
               [ 8.249159,  3.649159,  7.989159],
               [ 3.909159, -0.690841,  3.649159]])
        """
        if self._ci is not None and self._ci_cl is not None and (confidence_level == self._ci_cl):
            return self._ci
        if not 0 < confidence_level < 1:
            raise ValueError('Confidence level must be between 0 and 1.')
        params = (confidence_level, self._nobs, self._ntreatments - self._nobs)
        srd = distributions.studentized_range.ppf(*params)
        tukey_criterion = srd * self._stand_err
        upper_conf = self.statistic + tukey_criterion
        lower_conf = self.statistic - tukey_criterion
        self._ci = ConfidenceInterval(low=lower_conf, high=upper_conf)
        self._ci_cl = confidence_level
        return self._ci