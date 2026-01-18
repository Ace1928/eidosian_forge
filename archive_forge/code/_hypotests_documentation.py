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
Compute the confidence interval for the specified confidence level.

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
        