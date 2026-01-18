from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import warnings
import numpy as np
from scipy import special, interpolate, stats
from scipy.stats._censored_data import CensoredData
from scipy.stats._common import ConfidenceInterval
Compute a confidence interval around the CDF/SF point estimate

        Parameters
        ----------
        confidence_level : float, default: 0.95
            Confidence level for the computed confidence interval

        method : str, {"linear", "log-log"}
            Method used to compute the confidence interval. Options are
            "linear" for the conventional Greenwood confidence interval
            (default)  and "log-log" for the "exponential Greenwood",
            log-negative-log-transformed confidence interval.

        Returns
        -------
        ci : ``ConfidenceInterval``
            An object with attributes ``low`` and ``high``, instances of
            `~scipy.stats._result_classes.EmpiricalDistributionFunction` that
            represent the lower and upper bounds (respectively) of the
            confidence interval.

        Notes
        -----
        Confidence intervals are computed according to the Greenwood formula
        (``method='linear'``) or the more recent "exponential Greenwood"
        formula (``method='log-log'``) as described in [1]_. The conventional
        Greenwood formula can result in lower confidence limits less than 0
        and upper confidence limits greater than 1; these are clipped to the
        unit interval. NaNs may be produced by either method; these are
        features of the formulas.

        References
        ----------
        .. [1] Sawyer, Stanley. "The Greenwood and Exponential Greenwood
               Confidence Intervals in Survival Analysis."
               https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf

        