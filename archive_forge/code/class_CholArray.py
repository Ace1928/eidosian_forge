import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
class CholArray(PlainMatrixArray):
    """Class that defines linalg operation on an array

    cholesky version, where svd is taken on original data array, if
    or when it matters

    plan: use cholesky factor and cholesky solve
    nothing implemented yet
    """

    def __init__(self, data=None, sym=None):
        super(SvdArray, self).__init__(data=data, sym=sym)

    def yt_minv_y(self, y):
        """xSigmainvx
        does not use stored cholesky yet
        """
        return np.dot(x, linalg.cho_solve(linalg.cho_factor(self.m), x))