import numpy as np
from scipy._lib._util import _lazywhere
from scipy.special import gammaln
def _theta(mu, p):
    return np.where(p == 1, np.log(mu), mu ** (1 - p) / (1 - p))