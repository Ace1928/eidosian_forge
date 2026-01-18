import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less
from scipy import stats
import pytest
import statsmodels.genmod.families as families
from statsmodels.tools import numdiff as nd
class MyCLogLog(links.Link):

    def __call__(self, p):
        return np.log(-np.log(1 - p))

    def inverse(self, z):
        return 1 - np.exp(-np.exp(z))

    def deriv(self, p):
        return 1.0 / ((p - 1) * np.log(1 - p))