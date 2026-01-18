import numpy as np
from scipy.special import ndtri
from scipy.optimize import brentq
from ._discrete_distns import nchypergeom_fisher
from ._common import ConfidenceInterval
def _sample_odds_ratio_ci(self, confidence_level=0.95, alternative='two-sided'):
    """
        Confidence interval for the sample odds ratio.
        """
    if confidence_level < 0 or confidence_level > 1:
        raise ValueError('confidence_level must be between 0 and 1')
    table = self._table
    if 0 in table.sum(axis=0) or 0 in table.sum(axis=1):
        ci = (0, np.inf)
    else:
        ci = _sample_odds_ratio_ci(table, confidence_level=confidence_level, alternative=alternative)
    return ConfidenceInterval(low=ci[0], high=ci[1])