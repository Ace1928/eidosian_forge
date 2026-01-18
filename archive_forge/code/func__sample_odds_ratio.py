import numpy as np
from scipy.special import ndtri
from scipy.optimize import brentq
from ._discrete_distns import nchypergeom_fisher
from ._common import ConfidenceInterval
def _sample_odds_ratio(table):
    """
    Given a table [[a, b], [c, d]], compute a*d/(b*c).

    Return nan if the numerator and denominator are 0.
    Return inf if just the denominator is 0.
    """
    if table[1, 0] > 0 and table[0, 1] > 0:
        oddsratio = table[0, 0] * table[1, 1] / (table[1, 0] * table[0, 1])
    elif table[0, 0] == 0 or table[1, 1] == 0:
        oddsratio = np.nan
    else:
        oddsratio = np.inf
    return oddsratio