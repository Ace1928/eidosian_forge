import numpy as np
from scipy.optimize._slsqp import slsqp
from numpy import (zeros, array, linalg, append, concatenate, finfo,
from ._optimize import (OptimizeResult, _check_unknown_options,
from ._numdiff import approx_derivative
from ._constraints import old_bound_to_new, _arr_to_scalar
from scipy._lib._array_api import atleast_nd, array_namespace
from numpy import exp, inf  # noqa: F401
def _eval_con_normals(x, cons, la, n, m, meq, mieq):
    if cons['eq']:
        a_eq = vstack([con['jac'](x, *con['args']) for con in cons['eq']])
    else:
        a_eq = zeros((meq, n))
    if cons['ineq']:
        a_ieq = vstack([con['jac'](x, *con['args']) for con in cons['ineq']])
    else:
        a_ieq = zeros((mieq, n))
    if m == 0:
        a = zeros((la, n))
    else:
        a = vstack((a_eq, a_ieq))
    a = concatenate((a, zeros([la, 1])), 1)
    return a