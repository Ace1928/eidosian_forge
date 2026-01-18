import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from scipy import stats
from statsmodels.distributions.copula.api import (
from statsmodels.distributions.copula.api import transforms as tra
import statsmodels.distributions.tools as dt
from statsmodels.distributions.bernstein import (
class TestBernsteinBeta2dd(TestBernsteinBeta2d):

    @classmethod
    def setup_class(cls):
        grid = dt._Grid([91, 101])
        cop_tr = tra.TransfFrank
        args = (2,)
        ca = ArchimedeanCopula(cop_tr())
        distr1 = stats.beta(4, 3)
        distr2 = stats.beta(4, 4)
        cad = CopulaDistribution(ca, [distr1, distr2], cop_args=args)
        cdfv = cad.cdf(grid.x_flat, args)
        cdf_g = cdfv.reshape(grid.k_grid)
        cls.grid = grid
        cls.cdfv = cdfv
        cls.distr = cad
        cls.bpd = BernsteinDistribution(cdf_g)