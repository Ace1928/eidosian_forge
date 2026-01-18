import functools
from threading import RLock
import numpy as np
from scipy.optimize import _cobyla as cobyla
from ._optimize import (OptimizeResult, _check_unknown_options,
@synchronized
def _minimize_cobyla(fun, x0, args=(), constraints=(), rhobeg=1.0, tol=0.0001, maxiter=1000, disp=False, catol=0.0002, callback=None, bounds=None, **unknown_options):
    """
    Minimize a scalar function of one or more variables using the
    Constrained Optimization BY Linear Approximation (COBYLA) algorithm.

    Options
    -------
    rhobeg : float
        Reasonable initial changes to the variables.
    tol : float
        Final accuracy in the optimization (not precisely guaranteed).
        This is a lower bound on the size of the trust region.
    disp : bool
        Set to True to print convergence messages. If False,
        `verbosity` is ignored as set to 0.
    maxiter : int
        Maximum number of function evaluations.
    catol : float
        Tolerance (absolute) for constraint violations

    """
    _check_unknown_options(unknown_options)
    maxfun = maxiter
    rhoend = tol
    iprint = int(bool(disp))
    if isinstance(constraints, dict):
        constraints = (constraints,)
    if bounds:
        i_lb = np.isfinite(bounds.lb)
        if np.any(i_lb):

            def lb_constraint(x, *args, **kwargs):
                return x[i_lb] - bounds.lb[i_lb]
            constraints.append({'type': 'ineq', 'fun': lb_constraint})
        i_ub = np.isfinite(bounds.ub)
        if np.any(i_ub):

            def ub_constraint(x):
                return bounds.ub[i_ub] - x[i_ub]
            constraints.append({'type': 'ineq', 'fun': ub_constraint})
    for ic, con in enumerate(constraints):
        try:
            ctype = con['type'].lower()
        except KeyError as e:
            raise KeyError('Constraint %d has no type defined.' % ic) from e
        except TypeError as e:
            raise TypeError('Constraints must be defined using a dictionary.') from e
        except AttributeError as e:
            raise TypeError("Constraint's type must be a string.") from e
        else:
            if ctype != 'ineq':
                raise ValueError("Constraints of type '%s' not handled by COBYLA." % con['type'])
        if 'fun' not in con:
            raise KeyError('Constraint %d has no function defined.' % ic)
        if 'args' not in con:
            con['args'] = ()
    cons_lengths = []
    for c in constraints:
        f = c['fun'](x0, *c['args'])
        try:
            cons_length = len(f)
        except TypeError:
            cons_length = 1
        cons_lengths.append(cons_length)
    m = sum(cons_lengths)

    def _jac(x, *args):
        return None
    sf = _prepare_scalar_function(fun, x0, args=args, jac=_jac)

    def calcfc(x, con):
        f = sf.fun(x)
        i = 0
        for size, c in izip(cons_lengths, constraints):
            con[i:i + size] = c['fun'](x, *c['args'])
            i += size
        return f

    def wrapped_callback(x):
        if callback is not None:
            callback(np.copy(x))
    info = np.zeros(4, np.float64)
    xopt, info = cobyla.minimize(calcfc, m=m, x=np.copy(x0), rhobeg=rhobeg, rhoend=rhoend, iprint=iprint, maxfun=maxfun, dinfo=info, callback=wrapped_callback)
    if info[3] > catol:
        info[0] = 4
    return OptimizeResult(x=xopt, status=int(info[0]), success=info[0] == 1, message={1: 'Optimization terminated successfully.', 2: 'Maximum number of function evaluations has been exceeded.', 3: 'Rounding errors are becoming damaging in COBYLA subroutine.', 4: 'Did not converge to a solution satisfying the constraints. See `maxcv` for magnitude of violation.', 5: 'NaN result encountered.'}.get(info[0], 'Unknown exit status.'), nfev=int(info[1]), fun=info[2], maxcv=info[3])