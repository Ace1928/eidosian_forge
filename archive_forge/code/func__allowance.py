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