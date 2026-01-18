import numpy as np
import math
import inspect
import scipy.optimize
from scipy._lib._util import check_random_state
class MinimizerWrapper:
    """
    wrap a minimizer function as a minimizer class
    """

    def __init__(self, minimizer, func=None, **kwargs):
        self.minimizer = minimizer
        self.func = func
        self.kwargs = kwargs

    def __call__(self, x0):
        if self.func is None:
            return self.minimizer(x0, **self.kwargs)
        else:
            return self.minimizer(self.func, x0, **self.kwargs)