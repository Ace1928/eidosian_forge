from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def _eval_bspline_basis(x, knots, degree, deriv='all', include_intercept=True):
    try:
        from scipy.interpolate import splev
    except ImportError:
        raise ImportError('spline functionality requires scipy')
    knots = np.atleast_1d(np.asarray(knots, dtype=float))
    assert knots.ndim == 1
    knots.sort()
    degree = int(degree)
    x = np.atleast_1d(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    assert x.ndim == 1
    if np.min(x) < np.min(knots) or np.max(x) > np.max(knots):
        raise NotImplementedError("some data points fall outside the outermost knots, and I'm not sure how to handle them. (Patches accepted!)")
    k_const = 1 - int(include_intercept)
    n_bases = len(knots) - (degree + 1) - k_const
    if deriv in ['all', 0]:
        basis = np.empty((x.shape[0], n_bases), dtype=float)
        ret = basis
    if deriv in ['all', 1]:
        der1_basis = np.empty((x.shape[0], n_bases), dtype=float)
        ret = der1_basis
    if deriv in ['all', 2]:
        der2_basis = np.empty((x.shape[0], n_bases), dtype=float)
        ret = der2_basis
    for i in range(n_bases):
        coefs = np.zeros((n_bases + k_const,))
        coefs[i + k_const] = 1
        ii = i
        if deriv in ['all', 0]:
            basis[:, ii] = splev(x, (knots, coefs, degree))
        if deriv in ['all', 1]:
            der1_basis[:, ii] = splev(x, (knots, coefs, degree), der=1)
        if deriv in ['all', 2]:
            der2_basis[:, ii] = splev(x, (knots, coefs, degree), der=2)
    if deriv == 'all':
        return (basis, der1_basis, der2_basis)
    else:
        return ret