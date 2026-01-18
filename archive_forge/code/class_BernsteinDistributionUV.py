import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.distributions.tools import (
class BernsteinDistributionUV(BernsteinDistribution):

    def cdf(self, x, method='binom'):
        cdf_ = _eval_bernstein_1d(x, self.cdf_grid, method=method)
        return cdf_

    def pdf(self, x, method='binom'):
        pdf_ = self.k_grid_product * _eval_bernstein_1d(x, self.prob_grid, method=method)
        return pdf_