import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
class SvdArray(PlainMatrixArray):
    """Class that defines linalg operation on an array

    svd version, where svd is taken on original data array, if
    or when it matters

    no spectral cutoff in first version
    """

    def __init__(self, data=None, sym=None):
        super().__init__(data=data, sym=sym)
        u, s, v = np.linalg.svd(self.x, full_matrices=1)
        self.u, self.s, self.v = (u, s, v)
        self.sdiag = linalg.diagsvd(s, *x.shape)
        self.sinvdiag = linalg.diagsvd(1.0 / s, *x.shape)

    def _sdiagpow(self, p):
        return linalg.diagsvd(np.power(self.s, p), *x.shape)

    @cache_readonly
    def minv(self):
        sinvv = np.dot(self.sinvdiag, self.v)
        return np.dot(sinvv.T, sinvv)

    @cache_readonly
    def meigh(self):
        evecs = self.v.T
        evals = self.s ** 2
        return (evals, evecs)

    @cache_readonly
    def mdet(self):
        return self.meigh[0].prod()

    @cache_readonly
    def mlogdet(self):
        return np.log(self.meigh[0]).sum()

    @cache_readonly
    def mhalf(self):
        return np.dot(np.diag(self.s), self.v)

    @cache_readonly
    def xxthalf(self):
        return np.dot(self.u, self.sdiag)

    @cache_readonly
    def xxtinvhalf(self):
        return np.dot(self.u, self.sinvdiag)