import numpy as np
from scipy.special import ndtri
from scipy.optimize import brentq
from ._discrete_distns import nchypergeom_fisher
from ._common import ConfidenceInterval
def _conditional_odds_ratio_ci(self, confidence_level=0.95, alternative='two-sided'):
    """
        Confidence interval for the conditional odds ratio.
        """
    table = self._table
    if 0 in table.sum(axis=0) or 0 in table.sum(axis=1):
        ci = (0, np.inf)
    else:
        ci = _conditional_oddsratio_ci(table, confidence_level=confidence_level, alternative=alternative)
    return ConfidenceInterval(low=ci[0], high=ci[1])