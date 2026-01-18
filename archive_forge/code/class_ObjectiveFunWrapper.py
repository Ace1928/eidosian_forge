import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize, Bounds
from scipy.special import gammaln
from scipy._lib._util import check_random_state
from scipy.optimize._constraints import new_bounds_to_old
class ObjectiveFunWrapper:

    def __init__(self, func, maxfun=10000000.0, *args):
        self.func = func
        self.args = args
        self.nfev = 0
        self.ngev = 0
        self.nhev = 0
        self.maxfun = maxfun

    def fun(self, x):
        self.nfev += 1
        return self.func(x, *self.args)