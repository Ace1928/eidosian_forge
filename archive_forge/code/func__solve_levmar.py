from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def _solve_levmar(self, intern_x0, tol=1e-08, **kwargs):
    import warnings
    import levmar
    if 'eps1' in kwargs or 'eps2' in kwargs or 'eps3' in kwargs:
        pass
    else:
        kwargs['eps1'] = kwargs['eps2'] = kwargs['eps3'] = tol

    def _f(*args):
        return np.asarray(self.f_cb(*args))

    def _j(*args):
        return np.asarray(self.j_cb(*args))
    _x0 = np.asarray(intern_x0)
    _y0 = np.zeros(self.nf)
    with warnings.catch_warnings(record=True) as wrns:
        warnings.simplefilter('always')
        p_opt, p_cov, info = levmar.levmar(_f, _x0, _y0, args=(self.internal_params,), jacf=_j, **kwargs)
    success = len(wrns) == 0 and np.all(np.abs(_f(p_opt, self.internal_params)) < tol)
    for w in wrns:
        raise w
    e2p0, (e2, infJTe, Dp2, mu_maxJTJii), nit, reason, nfev, njev, nlinsolv = info
    return {'x': p_opt, 'cov': p_cov, 'nfev': nfev, 'njev': njev, 'nit': nit, 'message': reason, 'nlinsolv': nlinsolv, 'success': success}