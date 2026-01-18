import os
import numpy
from warnings import warn
from scipy.odr import __odrpack
def _cov2wt(self, cov):
    """ Convert covariance matrix(-ices) to weights.
        """
    from scipy.linalg import inv
    if len(cov.shape) == 2:
        return inv(cov)
    else:
        weights = numpy.zeros(cov.shape, float)
        for i in range(cov.shape[-1]):
            weights[:, :, i] = inv(cov[:, :, i])
        return weights