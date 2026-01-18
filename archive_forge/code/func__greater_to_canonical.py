import numpy as np
import scipy.sparse as sps
@classmethod
def _greater_to_canonical(cls, cfun, lb, keep_feasible):
    empty_fun = np.empty(0)
    n = cfun.n
    if cfun.sparse_jacobian:
        empty_jac = sps.csr_matrix((0, n))
    else:
        empty_jac = np.empty((0, n))
    finite_lb = lb > -np.inf
    n_eq = 0
    n_ineq = np.sum(finite_lb)
    if np.all(finite_lb):

        def fun(x):
            return (empty_fun, lb - cfun.fun(x))

        def jac(x):
            return (empty_jac, -cfun.jac(x))

        def hess(x, v_eq, v_ineq):
            return cfun.hess(x, -v_ineq)
    else:
        finite_lb = np.nonzero(finite_lb)[0]
        keep_feasible = keep_feasible[finite_lb]
        lb = lb[finite_lb]

        def fun(x):
            return (empty_fun, lb - cfun.fun(x)[finite_lb])

        def jac(x):
            return (empty_jac, -cfun.jac(x)[finite_lb])

        def hess(x, v_eq, v_ineq):
            v = np.zeros(cfun.m)
            v[finite_lb] = -v_ineq
            return cfun.hess(x, v)
    return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)