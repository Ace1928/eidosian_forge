import pytest
import numpy as np
from scipy.optimize import quadratic_assignment, OptimizeWarning
from scipy.optimize._qap import _calc_score as _score
from numpy.testing import assert_equal, assert_, assert_warns
def _doubly_stochastic(P, tol=0.001):
    max_iter = 1000
    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P
    for it in range(max_iter):
        if (np.abs(P_eps.sum(axis=1) - 1) < tol).all() and (np.abs(P_eps.sum(axis=0) - 1) < tol).all():
            break
        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c
    return P_eps