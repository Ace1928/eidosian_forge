import numpy as np
import scipy.sparse as sps
@classmethod
def _interval_to_canonical(cls, cfun, lb, ub, keep_feasible):
    lb_inf = lb == -np.inf
    ub_inf = ub == np.inf
    equal = lb == ub
    less = lb_inf & ~ub_inf
    greater = ub_inf & ~lb_inf
    interval = ~equal & ~lb_inf & ~ub_inf
    equal = np.nonzero(equal)[0]
    less = np.nonzero(less)[0]
    greater = np.nonzero(greater)[0]
    interval = np.nonzero(interval)[0]
    n_less = less.shape[0]
    n_greater = greater.shape[0]
    n_interval = interval.shape[0]
    n_ineq = n_less + n_greater + 2 * n_interval
    n_eq = equal.shape[0]
    keep_feasible = np.hstack((keep_feasible[less], keep_feasible[greater], keep_feasible[interval], keep_feasible[interval]))

    def fun(x):
        f = cfun.fun(x)
        eq = f[equal] - lb[equal]
        le = f[less] - ub[less]
        ge = lb[greater] - f[greater]
        il = f[interval] - ub[interval]
        ig = lb[interval] - f[interval]
        return (eq, np.hstack((le, ge, il, ig)))

    def jac(x):
        J = cfun.jac(x)
        eq = J[equal]
        le = J[less]
        ge = -J[greater]
        il = J[interval]
        ig = -il
        if sps.issparse(J):
            ineq = sps.vstack((le, ge, il, ig))
        else:
            ineq = np.vstack((le, ge, il, ig))
        return (eq, ineq)

    def hess(x, v_eq, v_ineq):
        n_start = 0
        v_l = v_ineq[n_start:n_start + n_less]
        n_start += n_less
        v_g = v_ineq[n_start:n_start + n_greater]
        n_start += n_greater
        v_il = v_ineq[n_start:n_start + n_interval]
        n_start += n_interval
        v_ig = v_ineq[n_start:n_start + n_interval]
        v = np.zeros_like(lb)
        v[equal] = v_eq
        v[less] = v_l
        v[greater] = -v_g
        v[interval] = v_il - v_ig
        return cfun.hess(x, v)
    return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)