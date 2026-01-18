import numpy as np
class UnivariateFanGijbels1EU(_UnivariateFunction):
    """

    Eubank p.179f
    """

    def __init__(self, nobs=50, x=None, distr_x=None, distr_noise=None):
        if distr_x is None:
            from scipy import stats
            distr_x = stats.uniform
        self.s_noise = 0.15
        self.func = fg1eu
        super(self.__class__, self).__init__(nobs=nobs, x=x, distr_x=distr_x, distr_noise=distr_noise)