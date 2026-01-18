from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def _solve_ipopt(self, intern_x0, **kwargs):
    import warnings
    from ipopt import minimize_ipopt
    warnings.warn('ipopt interface has not yet undergone thorough testing.')

    def f_cb(x):
        f_cb.nfev += 1
        return np.sum(np.abs(self.f_cb(x, self.internal_params)))
    f_cb.nfev = 0
    if self.j_cb is not None:

        def j_cb(x):
            j_cb.njev += 1
            return self.j_cb(x, self.internal_params)
        j_cb.njev = 0
        kwargs['jac'] = j_cb
    return minimize_ipopt(f_cb, intern_x0, **kwargs)