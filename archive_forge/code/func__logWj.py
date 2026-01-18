import numpy as np
from scipy._lib._util import _lazywhere
from scipy.special import gammaln
def _logWj(y, j, p, phi):
    alpha = _alpha(p)
    logz = -alpha * np.log(y) + alpha * np.log(p - 1) - (1 - alpha) * np.log(phi) - np.log(2 - p)
    return j * logz - gammaln(1 + j) - gammaln(-alpha * j)