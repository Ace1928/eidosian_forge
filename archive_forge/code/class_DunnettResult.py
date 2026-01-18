from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._stats_py import _var
@dataclass
class DunnettResult:
    """Result object returned by `scipy.stats.dunnett`.

    Attributes
    ----------
    statistic : float ndarray
        The computed statistic of the test for each comparison. The element
        at index ``i`` is the statistic for the comparison between
        groups ``i`` and the control.
    pvalue : float ndarray
        The computed p-value of the test for each comparison. The element
        at index ``i`` is the p-value for the comparison between
        group ``i`` and the control.
    """
    statistic: np.ndarray
    pvalue: np.ndarray
    _alternative: Literal['two-sided', 'less', 'greater'] = field(repr=False)
    _rho: np.ndarray = field(repr=False)
    _df: int = field(repr=False)
    _std: float = field(repr=False)
    _mean_samples: np.ndarray = field(repr=False)
    _mean_control: np.ndarray = field(repr=False)
    _n_samples: np.ndarray = field(repr=False)
    _n_control: int = field(repr=False)
    _rng: SeedType = field(repr=False)
    _ci: ConfidenceInterval | None = field(default=None, repr=False)
    _ci_cl: DecimalNumber | None = field(default=None, repr=False)

    def __str__(self):
        if self._ci is None:
            self.confidence_interval(confidence_level=0.95)
        s = f"Dunnett's test ({self._ci_cl * 100:.1f}% Confidence Interval)\nComparison               Statistic  p-value  Lower CI  Upper CI\n"
        for i in range(self.pvalue.size):
            s += f' (Sample {i} - Control) {self.statistic[i]:>10.3f}{self.pvalue[i]:>10.3f}{self._ci.low[i]:>10.3f}{self._ci.high[i]:>10.3f}\n'
        return s

    def _allowance(self, confidence_level: DecimalNumber=0.95, tol: DecimalNumber=0.001) -> float:
        """Allowance.

        It is the quantity to add/subtract from the observed difference
        between the means of observed groups and the mean of the control
        group. The result gives confidence limits.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval.
            Default is .95.
        tol : float, optional
            A tolerance for numerical optimization: the allowance will produce
            a confidence within ``10*tol*(1 - confidence_level)`` of the
            specified level, or a warning will be emitted. Tight tolerances
            may be impractical due to noisy evaluation of the objective.
            Default is 1e-3.

        Returns
        -------
        allowance : float
            Allowance around the mean.
        """
        alpha = 1 - confidence_level

        def pvalue_from_stat(statistic):
            statistic = np.array(statistic)
            sf = _pvalue_dunnett(rho=self._rho, df=self._df, statistic=statistic, alternative=self._alternative, rng=self._rng)
            return abs(sf - alpha) / alpha
        res = minimize_scalar(pvalue_from_stat, method='brent', tol=tol)
        critical_value = res.x
        if res.success is False or res.fun >= tol * 10:
            warnings.warn(f'Computation of the confidence interval did not converge to the desired level. The confidence level corresponding with the returned interval is approximately {alpha * (1 + res.fun)}.', stacklevel=3)
        allowance = critical_value * self._std * np.sqrt(1 / self._n_samples + 1 / self._n_control)
        return abs(allowance)

    def confidence_interval(self, confidence_level: DecimalNumber=0.95) -> ConfidenceInterval:
        """Compute the confidence interval for the specified confidence level.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval.
            Default is .95.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence intervals for each
            comparison. The high and low values are accessible for each
            comparison at index ``i`` for each group ``i``.

        """
        if self._ci is not None and confidence_level == self._ci_cl:
            return self._ci
        if not 0 < confidence_level < 1:
            raise ValueError('Confidence level must be between 0 and 1.')
        allowance = self._allowance(confidence_level=confidence_level)
        diff_means = self._mean_samples - self._mean_control
        low = diff_means - allowance
        high = diff_means + allowance
        if self._alternative == 'greater':
            high = [np.inf] * len(diff_means)
        elif self._alternative == 'less':
            low = [-np.inf] * len(diff_means)
        self._ci_cl = confidence_level
        self._ci = ConfidenceInterval(low=low, high=high)
        return self._ci