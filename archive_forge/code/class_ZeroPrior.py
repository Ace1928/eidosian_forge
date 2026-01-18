import numpy as np
class ZeroPrior(Prior):
    """ZeroPrior object, consisting on a constant prior with 0eV energy."""

    def __init__(self):
        Prior.__init__(self)

    def potential(self, x):
        return np.zeros(x.shape[0] + 1)