import numpy as np
class L2Univariate(Penalty):
    """
    The L2 (ridge) penalty applied to each parameter.
    """

    def __init__(self, weights=None):
        if weights is None:
            self.weights = 1.0
        else:
            self.weights = weights

    def func(self, params):
        return self.weights * params ** 2

    def deriv(self, params):
        return 2 * self.weights * params

    def deriv2(self, params):
        return 2 * self.weights * np.ones(len(params))